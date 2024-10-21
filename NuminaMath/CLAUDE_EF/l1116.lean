import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_ones_largest_number_proof_l1116_111645

def four_ones_largest_number : ℕ := 11^11

theorem four_ones_largest_number_proof :
  ∀ n : ℕ, (n.repr.count '1' = 4 ∧ n.repr.all (λ c => c = '1')) →
  n ≤ four_ones_largest_number :=
by
  sorry

#eval four_ones_largest_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_ones_largest_number_proof_l1116_111645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_POQ_l1116_111698

noncomputable section

-- Define the unit circle
def unit_circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define points P and Q
noncomputable def P : ℝ × ℝ := (4/5, 3/5)
noncomputable def Q : ℝ × ℝ := (5/13, -12/13)

-- State the theorem
theorem cos_angle_POQ :
  P ∈ unit_circle ∧ 
  Q ∈ unit_circle ∧ 
  P.1 > 0 ∧ P.2 > 0 ∧  -- P is in the first quadrant
  Q.1 > 0 ∧ Q.2 < 0 →  -- Q is in the fourth quadrant
  Real.arccos ((P.1 * Q.1 + P.2 * Q.2) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt (Q.1^2 + Q.2^2))) = Real.arccos (56/65) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_POQ_l1116_111698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l1116_111624

theorem simplify_fraction (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 1) :
  (1 - a / (a + 1)) / (1 / (1 - a^2)) = (1 - a) * (1 + a) / (a + 1) := by
  sorry

#check simplify_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l1116_111624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_half_l1116_111664

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem: The eccentricity of a specific ellipse -/
theorem ellipse_eccentricity_is_half :
  ∃ (k : ℝ) (e : Ellipse), 
    k > -1 ∧
    e.a^2 = k + 2 ∧
    e.b^2 = k + 1 ∧
    (∃ (p : ℝ), p = 8 ∧ p = 4 * e.a) →
    e.eccentricity = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_half_l1116_111664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1116_111652

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 1

theorem f_symmetry : ∀ x : ℝ, f (Real.pi / 6 + x) = f (Real.pi / 6 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1116_111652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_l1116_111607

theorem cos_two_pi_thirds : Real.cos (2 * Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_l1116_111607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_orthogonal_chord_constant_distance_l1116_111625

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  k : ℝ
  m : ℝ

/-- The ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- Check if two vectors are perpendicular -/
def perpendicular (a b : Point) : Prop :=
  a.x * b.x + a.y * b.y = 0

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (l : Line) : ℝ :=
  |l.m| / Real.sqrt (l.k^2 + 1)

/-- Main theorem -/
theorem ellipse_orthogonal_chord_constant_distance :
  ∀ (A B : Point) (l : Line),
    ellipse_C A ∧ ellipse_C B ∧
    (∃ (k m : ℝ), A.y = k * A.x + m ∧ B.y = k * B.x + m) ∧
    perpendicular A B →
    distance_point_to_line l = 2 * Real.sqrt 21 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_orthogonal_chord_constant_distance_l1116_111625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_and_stability_l1116_111620

-- Define the system parameters
variable (m : ℝ) -- distance between pulleys
variable (r : ℝ) -- radius of the wire ring
variable (G : ℝ) -- weight
variable (Q : ℝ) -- tension in the string

-- Define the angle α
variable (α : ℝ)

-- Define the length of the string segment
noncomputable def l (m r α : ℝ) : ℝ := Real.sqrt ((m - r)^2 + r^2 - 2*r*(m-r)*Real.cos α)

-- Equilibrium condition
def equilibrium_condition (m r G Q α : ℝ) : Prop :=
  Real.cos α = ((m-r)^2 * (1 - (Q^2 / G^2)) + r^2) / (2*r*(m-r))

-- Physical solution condition
def physical_solution (m r G Q : ℝ) : Prop :=
  m * G > (m - r) * Q

-- Potential energy function
noncomputable def potential_energy (G Q m r α : ℝ) : ℝ :=
  -G*r*(1 - Real.cos α) + Q*(Real.sqrt ((m-r)^2 + r^2 - 2*r*(m-r)*Real.cos α) - m + 2*r)

-- Stability conditions
def stability_condition_A (m r G Q : ℝ) : Prop :=
  G < ((m - r) / (m - 2*r)) * Q

def stability_condition_B (m r G Q : ℝ) : Prop :=
  G > ((m - r) / m) * Q

-- Theorem statement
theorem equilibrium_and_stability 
  (m r G Q : ℝ) 
  (h1 : m > 0) 
  (h2 : r > 0) 
  (h3 : G > 0) 
  (h4 : Q > 0) 
  (h5 : m > r) :
  (∃ α, equilibrium_condition m r G Q α ∧ 
        physical_solution m r G Q) ∧
  (stability_condition_A m r G Q ∨ 
   stability_condition_B m r G Q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_and_stability_l1116_111620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_purchase_l1116_111627

/-- Calculates the total number of items purchased given a budget and item costs --/
def totalItems (budget : ℚ) (pieCost : ℚ) (juiceCost : ℚ) : ℕ :=
  let numPies := (budget / pieCost).floor.toNat
  let remainingBudget := budget - numPies * pieCost
  let numJuices := (remainingBudget / juiceCost).floor.toNat
  numPies + numJuices

/-- Proves that given the specified conditions, the total number of items purchased is 9 --/
theorem bakery_purchase : totalItems 50 6 (3/2) = 9 := by
  sorry

#eval totalItems 50 6 (3/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_purchase_l1116_111627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l1116_111644

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x^2 - 4)

theorem one_vertical_asymptote :
  ∃! a : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l1116_111644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l1116_111649

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse (a b c : ℝ) : Type :=
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2)

/-- A point on the ellipse. -/
structure PointOnEllipse (E : Ellipse a b c) : Type :=
  (x y : ℝ)
  (on_ellipse : x^2 / a^2 + y^2 / b^2 = 1)

/-- Intersection point of a line and the ellipse -/
def IntersectionPoint (E : Ellipse a b c) := ℝ × ℝ

theorem ellipse_ratio_sum (a b c : ℝ) (E : Ellipse a b c) 
  (h_arithmetic : a^2 - b^2 = b^2 - c^2) 
  (P : PointOnEllipse E) :
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let A : IntersectionPoint E := sorry -- The intersection of PF₁ with the ellipse
  let B : IntersectionPoint E := sorry -- The intersection of PF₂ with the ellipse
  ∃ (PF₁ AF₁ PF₂ BF₂ : ℝ), 
    PF₁ / AF₁ + PF₂ / BF₂ = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l1116_111649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l1116_111656

/-- Daily cost function -/
noncomputable def C (x : ℝ) : ℝ := 3 + x

/-- Daily sales revenue function -/
noncomputable def S (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 6 then 3*x + k/(x-8) + 5 else 14

/-- Daily profit function -/
noncomputable def L (k : ℝ) (x : ℝ) : ℝ := S k x - C x

theorem profit_maximization (k : ℝ) :
  (L k 2 = 3) →
  (∃ k', k = k' ∧
    (∀ x, L k' x ≤ 6) ∧
    (L k' 5 = 6) ∧
    (∀ x, 0 < x → x < 6 → L k' x ≤ L k' 5) ∧
    (∀ x, x ≥ 6 → L k' x ≤ L k' 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l1116_111656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_35_l1116_111634

/-- The coefficient of x^2 in the expansion of (1+x) + (1+x)^2 + ... + (1+x)^6 -/
def coefficient_x_squared : ℕ :=
  (List.range 6).map (fun k => Nat.choose (k + 1) 2) |> List.sum

/-- Theorem stating that the coefficient of x^2 in the expansion is 35 -/
theorem coefficient_x_squared_is_35 : coefficient_x_squared = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_35_l1116_111634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_probability_theorem_l1116_111673

/-- The probability of snowing on any given day in December in Frost Town -/
noncomputable def snow_prob : ℝ := 1/5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The probability of snowing on at most 3 days in December in Frost Town -/
noncomputable def at_most_three_snow_days : ℝ :=
  (1 - snow_prob)^december_days +
  (december_days.choose 1) * snow_prob * (1 - snow_prob)^(december_days - 1) +
  (december_days.choose 2) * snow_prob^2 * (1 - snow_prob)^(december_days - 2) +
  (december_days.choose 3) * snow_prob^3 * (1 - snow_prob)^(december_days - 3)

/-- The theorem stating that the probability of snowing on at most 3 days in December
    in Frost Town is approximately 0.230 -/
theorem snow_probability_theorem :
  abs (at_most_three_snow_days - 0.230) < 0.0005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_probability_theorem_l1116_111673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1116_111697

theorem negation_of_universal_proposition :
  (¬∀ x : ℝ, Real.cos x > Real.sin x - 1) ↔ (∃ x : ℝ, Real.cos x ≤ Real.sin x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1116_111697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_circle_l1116_111694

/-- The parabola y = x² -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- The circle (x-4)² + (y+1/2)² = 1 -/
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y + 1/2)^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The minimum distance between a point on the parabola and a point on the circle -/
theorem min_distance_parabola_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), parabola x₃ y₃ → circle_eq x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = 3 * Real.sqrt 5 / 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_circle_l1116_111694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l1116_111696

/-- Given x and y values, and a linear relationship between them, prove the value of a -/
theorem find_a_value (x : Fin 4 → ℝ) (y : Fin 4 → ℝ) 
  (hx : x = ![0, 1, 3, 4])
  (hy : y = ![a, 4.3, 4.8, 6.7])
  (h_linear : ∀ i : Fin 4, y i = 0.95 * (x i) + 2.6)
  (a : ℝ) : a = 2.2 := by
  sorry

#check find_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l1116_111696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_revenue_fraction_l1116_111675

theorem lemonade_revenue_fraction (total_cups : ℚ) (small_price : ℚ) : 
  total_cups > 0 → small_price > 0 →
  (let small_cups := (1 : ℚ) / 8 * total_cups
   let medium_cups := (3 : ℚ) / 8 * total_cups
   let large_cups := total_cups - small_cups - medium_cups
   let medium_price := (3 : ℚ) / 2 * small_price
   let large_price := (5 : ℚ) / 2 * small_price
   let small_revenue := small_cups * small_price
   let medium_revenue := medium_cups * medium_price
   let large_revenue := large_cups * large_price
   let total_revenue := small_revenue + medium_revenue + large_revenue
   large_revenue / total_revenue = 20 / 31) := by
  intro h_total_cups h_small_price
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_revenue_fraction_l1116_111675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_matches_l1116_111699

/-- Represents the number of teams in the tournament -/
def x : ℕ := sorry

/-- The total number of matches in the tournament -/
def total_matches : ℕ := 21

/-- The number of matches in a single round-robin tournament with x teams -/
def matches_formula (x : ℕ) : ℚ := (1 / 2) * x * (x - 1)

/-- Theorem stating that the number of matches in a single round-robin tournament 
    with x teams is equal to the planned total matches -/
theorem round_robin_matches : 
  matches_formula x = total_matches := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_matches_l1116_111699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_paper_selection_l1116_111636

/-- A cell on a graph paper -/
structure Cell where
  x : ℤ
  y : ℤ

/-- Two cells are adjacent if they share a side or a corner -/
def adjacent (c1 c2 : Cell) : Prop :=
  (abs (c1.x - c2.x) ≤ 1) ∧ (abs (c1.y - c2.y) ≤ 1)

/-- A set of cells is non-intersecting if no two cells in the set are adjacent -/
def non_intersecting (s : Set Cell) : Prop :=
  ∀ c1 c2, c1 ∈ s → c2 ∈ s → c1 ≠ c2 → ¬adjacent c1 c2

theorem graph_paper_selection (n : ℕ) (cells : Finset Cell) (h : cells.card = n) :
  ∃ s : Finset Cell, s ⊆ cells ∧ non_intersecting s ∧ s.card ≥ n / 4 := by
  sorry

#check graph_paper_selection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_paper_selection_l1116_111636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l1116_111621

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := ((2 * x + 6) / 4) ^ (1/3)

-- State the theorem
theorem g_equality (x : ℝ) : g (3 * x) = 3 * g x ↔ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l1116_111621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dalmatian_vote_theorem_l1116_111612

/-- The number of Dalmatians participating in the election -/
def num_dalmatians : ℕ := 101

/-- The probability of a Dalmatian voting for either candidate -/
def vote_probability : ℚ := 1/2

/-- X represents the number of Dalmatians who voted for the winning candidate -/
def X : ℕ → ℕ := sorry

/-- The expected value of X^2 -/
noncomputable def E_X_squared : ℚ := sorry

/-- The numerator of E[X^2] when expressed as a fraction a/b -/
def a : ℕ := sorry

/-- The denominator of E[X^2] when expressed as a fraction a/b -/
def b : ℕ := sorry

/-- Assumption that a and b are coprime -/
axiom a_b_coprime : Nat.Coprime a b

/-- The unique positive integer k ≤ 103 such that 103 | a - bk -/
def k : ℕ := sorry

/-- Main theorem stating the properties of k -/
theorem dalmatian_vote_theorem :
  k ≤ 103 ∧
  (∃ m : ℤ, a - b * k = 103 * m) ∧
  (∀ k' : ℕ, k' ≤ 103 → k' ≠ k → ¬(∃ m : ℤ, a - b * k' = 103 * m)) ∧
  k = 51 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dalmatian_vote_theorem_l1116_111612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_51_l1116_111647

def g (x : Int) : Int := x^2 - 3*x + 2023

theorem gcd_g_50_51 : Int.gcd (g 50) (g 51) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_51_l1116_111647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l1116_111679

/-- Crystal's running course -/
structure RunningCourse where
  north : ℝ
  northeast : ℝ
  southeast : ℝ
  west : ℝ

/-- Calculate the final distance of Crystal's run -/
noncomputable def final_distance (course : RunningCourse) : ℝ :=
  let east_west := course.northeast * (Real.sqrt 2) / 2 + course.southeast * (Real.sqrt 2) / 2 - course.west
  let north_south := course.north + course.northeast * (Real.sqrt 2) / 2 - course.southeast * (Real.sqrt 2) / 2
  Real.sqrt (east_west ^ 2 + north_south ^ 2)

/-- Crystal's specific running course -/
def crystal_course : RunningCourse :=
  { north := 2
  , northeast := 2
  , southeast := 3
  , west := 2 }

/-- Theorem: The final distance of Crystal's run is equal to the given expression -/
theorem crystal_run_distance :
  final_distance crystal_course = Real.sqrt ((5 * Real.sqrt 2 - 4)^2 / 4 + (4 - Real.sqrt 2)^2 / 4) := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l1116_111679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1116_111648

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f (Real.pi / 3 + x) = -f (Real.pi / 3 - x)) ∧
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x₀ + 2 * Real.sqrt 3 ≤ f x + 2 * Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x + 2 * Real.sqrt 3 ≥ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1116_111648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_for_nth_mile_correct_l1116_111683

/-- Represents the time (in seconds) needed to traverse the nth mile -/
noncomputable def time_for_nth_mile (n : ℕ) : ℝ :=
  (27 * (n - 1)^2 / 4) ^ (1/3)

/-- Represents the speed (in miles per second) for the nth mile -/
noncomputable def speed_for_nth_mile (n : ℕ) (t : ℝ) : ℝ :=
  (4 * t^2) / (27 * (n - 1)^2)

theorem time_for_nth_mile_correct (n : ℕ) (hn : n ≥ 3) :
  let t := time_for_nth_mile n
  speed_for_nth_mile n t = 1 / t ∧ 
  (n = 3 → t = 3) := by
  sorry

#check time_for_nth_mile_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_for_nth_mile_correct_l1116_111683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_upper_bound_l1116_111641

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ k : ℕ, k ∈ (Set.range a) ∨ ∃! (i j : ℕ), i < j ∧ k = a i + a j)

theorem sequence_upper_bound (a : ℕ → ℕ) (h : is_valid_sequence a) :
  ∀ n : ℕ, a n < n^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_upper_bound_l1116_111641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_pile_volume_is_57_6_pi_l1116_111689

/-- The volume of a conical pile of gravel -/
noncomputable def gravel_pile_volume (diameter : ℝ) (height_ratio : ℝ) : ℝ :=
  let radius := diameter / 2
  let height := height_ratio * diameter
  (1 / 3) * Real.pi * radius^2 * height

/-- Theorem: The volume of the specified gravel pile is 57.6π cubic feet -/
theorem gravel_pile_volume_is_57_6_pi :
  gravel_pile_volume 12 0.4 = 57.6 * Real.pi := by
  sorry

#check gravel_pile_volume_is_57_6_pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_pile_volume_is_57_6_pi_l1116_111689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_effect_l1116_111661

theorem price_reduction_effect (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  (0.8 * P * 1.8 * Q - P * Q) / (P * Q) = 0.44 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_effect_l1116_111661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l1116_111692

/-- Given a real number a and a function f(x) = x³ + ax where x ∈ ℝ,
    if f has an extremum at x = 1, then the equation of the tangent line
    to the curve y = f(x) at the origin is y = -3x. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  (λ x ↦ -3*x) = λ x ↦ (deriv f 0) * x :=
by
  intro f extremum_condition
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l1116_111692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_product_condition_point_p_values_l1116_111604

/-- Polynomial A -/
def A (x m : ℝ) : ℝ := 2 * x^2 - m * x + 1

/-- Polynomial B -/
def B (x n : ℝ) : ℝ := n * x^2 - 3

/-- The product of polynomials A and B -/
def product (x m n : ℝ) : ℝ := A x m * B x n

/-- Theorem stating the conditions for m and n -/
theorem polynomial_product_condition (m n : ℝ) :
  (∀ x, product x m n = 2 * n * x^4 + 3 * m * x - 3) →
  m = 0 ∧ n = 6 :=
sorry

/-- Function representing the distance condition for point P -/
def distance_condition (p : ℝ) : Prop :=
  |p - 0| = 2 * |p - 6|

/-- Theorem stating the possible values for point P -/
theorem point_p_values :
  ∃ p, distance_condition p ↔ p = 12 ∨ p = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_product_condition_point_p_values_l1116_111604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_min_point_l1116_111667

-- Define the original function
def f (x : ℝ) : ℝ := abs (x + 1) - 4

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - 3) + 4

-- Theorem stating that the minimum point of g is (2, 0)
theorem translated_min_point :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), g x₀ ≤ g x) ∧ (x₀ = 2) ∧ (g x₀ = 0) := by
  sorry

#check translated_min_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_min_point_l1116_111667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_ab_l1116_111678

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line a^2x + y + 2 = 0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a^2

/-- The slope of the line bx - (a^2 + 1)y - 1 = 0 -/
noncomputable def slope2 (a b : ℝ) : ℝ := b / (a^2 + 1)

/-- The minimum value of |ab| given perpendicular lines -/
theorem min_abs_ab (a b : ℝ) : 
  perpendicular (slope1 a) (slope2 a b) → 
  ∃ (min : ℝ), min = 1 ∧ ∀ (x y : ℝ), perpendicular (slope1 x) (slope2 x y) → min ≤ |x*y| :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_ab_l1116_111678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_cost_l1116_111677

/-- Represents the amusement park cost model -/
structure AmusementParkCost where
  initialCost : ℝ        -- Initial construction cost
  annualFixedCost : ℝ    -- Annual fixed cost
  initialMaintenance : ℝ -- First year maintenance cost
  maintenanceIncrease : ℝ -- Annual increase in maintenance cost

/-- Calculates the average cost per year for operating the amusement park -/
noncomputable def averageCost (c : AmusementParkCost) (years : ℝ) : ℝ :=
  (c.initialCost + c.annualFixedCost * years + 
   c.initialMaintenance * years + c.maintenanceIncrease * (years - 1) * years / 2) / years

/-- Theorem: The average cost is minimized when the park operates for 10 years -/
theorem min_average_cost (c : AmusementParkCost) 
  (h1 : c.initialCost = 500000)
  (h2 : c.annualFixedCost = 45000)
  (h3 : c.initialMaintenance = 10000)
  (h4 : c.maintenanceIncrease = 10000) :
  ∃ (y : ℝ), y > 0 ∧ ∀ (x : ℝ), x > 0 → averageCost c y ≤ averageCost c x ∧ y = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_cost_l1116_111677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1116_111659

/-- An ellipse with semi-major axis 2 and semi-minor axis b -/
structure Ellipse where
  b : ℝ
  h_pos : b > 0

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2/4 + y^2/e.b^2 = 1

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse) : ℝ × ℝ := sorry

/-- A point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  on_ellipse : e.equation x y

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The maximum sum of distances from two points on the ellipse to the right focus -/
noncomputable def max_sum_distances (e : Ellipse) : ℝ := sorry

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := sorry

theorem ellipse_eccentricity (e : Ellipse) :
  max_sum_distances e = 5 → eccentricity e = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1116_111659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_and_sum_l1116_111680

open Real BigOperators

-- Define the series term
noncomputable def a (n : ℕ) : ℝ := 18 / (n^2 + n - 2)

-- State the theorem
theorem series_convergence_and_sum :
  (∑' n, a (n + 2)) = 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_and_sum_l1116_111680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_solutions_l1116_111657

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the domain of x
def domain : Set ℝ := Set.Ioo 0 2

-- Define the equation
noncomputable def equation (m x : ℝ) : ℝ := |g x|^2 + m * |g x| + 2 * m + 3

-- Theorem statement
theorem three_distinct_solutions (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ domain ∧ x₂ ∈ domain ∧ x₃ ∈ domain ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    equation m x₁ = 0 ∧ equation m x₂ = 0 ∧ equation m x₃ = 0) ↔
  m ∈ Set.Ioo (-3/2) (-4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_solutions_l1116_111657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l1116_111613

/-- Represents the work scenario with original and new conditions -/
structure WorkScenario where
  original_men : ℕ
  original_hours_per_day : ℕ
  original_days : ℕ
  new_men : ℕ
  new_days : ℕ

/-- Calculates the new hours per day required to complete the work -/
def new_hours_per_day (scenario : WorkScenario) : ℚ :=
  (scenario.original_men * scenario.original_hours_per_day : ℚ) / scenario.new_men * scenario.new_days

/-- Theorem stating the relationship between original and new work conditions -/
theorem work_completion_theorem (scenario : WorkScenario) 
    (h1 : scenario.original_hours_per_day = 9)
    (h2 : scenario.original_days = 24)
    (h3 : scenario.new_men = 12)
    (h4 : scenario.new_days = 16) :
  new_hours_per_day scenario = (scenario.original_men * 9 : ℚ) / 8 := by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l1116_111613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1116_111609

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define the line equation
def line (p : PolarPoint) : Prop :=
  4 * p.ρ * Real.cos (p.θ - Real.pi/6) + 1 = 0

-- Define the circle equation
def circleEq (p : PolarPoint) : Prop :=
  p.ρ = 2 * Real.sin p.θ

-- Define the number of common points
def common_points : ℕ := 2

-- Theorem statement
theorem line_circle_intersection :
  ∃ (S : Finset PolarPoint),
    (∀ p ∈ S, line p ∧ circleEq p) ∧
    S.card = common_points := by
  sorry

#check line_circle_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1116_111609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_transaction_profit_l1116_111601

/-- Calculates the net profit for Mr. A in a series of house transactions --/
theorem house_transaction_profit 
  (initial_value : ℝ) 
  (maintenance_cost : ℝ) 
  (first_sale_profit_percent : ℝ) 
  (renovation_cost : ℝ) 
  (second_sale_loss_percent : ℝ) 
  (h1 : initial_value = 20000)
  (h2 : maintenance_cost = 2000)
  (h3 : first_sale_profit_percent = 0.2)
  (h4 : renovation_cost = 3000)
  (h5 : second_sale_loss_percent = 0.15) :
  (initial_value + maintenance_cost) * (1 + first_sale_profit_percent) -
  ((initial_value + maintenance_cost) * (1 + first_sale_profit_percent) + renovation_cost) * (1 - second_sale_loss_percent) = 1410 :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_transaction_profit_l1116_111601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_2_l1116_111650

/-- The function f(x) with parameters ω and b -/
noncomputable def f (ω : ℝ) (b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

/-- The period of the function f(x) -/
noncomputable def T (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem f_value_at_pi_over_2 (ω : ℝ) (b : ℝ) :
  ω > 0 →
  2 * Real.pi / 3 < T ω →
  T ω < Real.pi →
  (∀ x : ℝ, f ω b (3 * Real.pi / 2 - x) = f ω b (3 * Real.pi / 2 + x)) →
  f ω b (3 * Real.pi / 2) = 2 →
  f ω b (Real.pi / 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_2_l1116_111650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1116_111654

-- Define the triangle ABC
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- State the theorem
theorem triangle_properties 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle a b c A B C)
  (h_equation : Real.sqrt 3 * a * Real.sin B - b * Real.cos A = b)
  (h_sum : b + c = 4) :
  A = Real.pi / 3 ∧ 
  (∃ (a_min : ℝ), ∀ (a' : ℝ), triangle a' b c A B C → a' ≥ a_min ∧ 
    (a' = a_min → (1/2) * b * c * Real.sin A = Real.sqrt 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1116_111654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primes_from_30_l1116_111605

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (fun d => if d > 1 then n % d ≠ 0 else true)

def count_primes (n : ℕ) : ℕ :=
  (List.range n).filter is_prime |>.length

theorem probability_two_primes_from_30 :
  let total_choices := (30 * 29) / 2
  let prime_choices := (count_primes 31 * (count_primes 31 - 1)) / 2
  (prime_choices : ℚ) / total_choices = 3 / 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primes_from_30_l1116_111605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_neg_one_l1116_111638

/-- The function g satisfies the given functional equation for all x ≠ 2/3 -/
axiom g : ℝ → ℝ

/-- The functional equation for g -/
axiom g_equation (x : ℝ) (h : x ≠ 2/3) : g x + g ((x + 2) / (2 - 3*x)) = 2*x

/-- The value of g at -1 -/
theorem g_at_neg_one : g (-1 : ℝ) = -61/65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_neg_one_l1116_111638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l1116_111622

def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i ↦ a i)

theorem special_sequence_sum :
  ∀ a : ℕ → ℝ,
  (∀ n, arithmetic_sequence 2 (sequence_sum a n) (3 * a n)) →
  sequence_sum a 5 = -242 :=
by
  sorry

#check special_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l1116_111622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1116_111640

/-- The value of k for which |z| = k intersects |z - 4| = 3|z + 2| at exactly one point -/
noncomputable def k : ℝ := 2.75 + Real.sqrt (137 / 32)

/-- The locus of points satisfying |z - 4| = 3|z + 2| -/
def locus (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 2)

/-- Theorem stating that there exists a unique point satisfying both conditions -/
theorem unique_intersection :
  ∃! (z : ℂ), locus z ∧ Complex.abs z = k := by
  sorry

#check unique_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1116_111640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_theorem_l1116_111608

/-- Represents the outcome of rolling an eight-sided die -/
inductive DieRoll
| one | two | three | four | five | six | seven | eight

/-- Defines the behavior of Alice based on the die roll -/
def roll_behavior (r : DieRoll) : Bool :=
  match r with
  | DieRoll.two | DieRoll.three | DieRoll.four | DieRoll.seven => true  -- Drinks coffee
  | DieRoll.eight => false  -- Rolls again
  | _ => false  -- Drinks tea

/-- The probability of each outcome on a fair eight-sided die -/
def roll_probability : DieRoll → ℚ
| _ => 1/8

/-- The expected number of rolls on a single day -/
noncomputable def expected_rolls_per_day : ℚ := 6/7

/-- The number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
noncomputable def expected_rolls_in_year : ℚ := expected_rolls_per_day * days_in_year

theorem expected_rolls_theorem : 
  expected_rolls_in_year = 313.57142857142856 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_theorem_l1116_111608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_constant_range_of_f_l1116_111646

/-- The function f as defined in the problem -/
noncomputable def f (x a b c : ℝ) : ℝ :=
  (x + a)^2 / ((a - b) * (a - c)) +
  (x + b)^2 / ((b - a) * (b - c)) +
  (x + c)^2 / ((c - a) * (c - b))

/-- Theorem stating that f(x) is always 1 for distinct a, b, c -/
theorem f_is_constant (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  f x a b c = 1 := by
  sorry

/-- Corollary: The range of f is {1} -/
theorem range_of_f (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  Set.range (λ x ↦ f x a b c) = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_constant_range_of_f_l1116_111646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_squared_l1116_111688

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Defines the foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : (Point × Point) :=
  let e := Real.sqrt (h.a^2 + h.b^2) / h.a
  ({ x := e * h.a, y := 0 }, { x := -e * h.a, y := 0 })

/-- Checks if three points form an isosceles right triangle -/
noncomputable def is_isosceles_right_triangle (p1 p2 p3 : Point) : Prop :=
  let d12 := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := Real.sqrt ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := Real.sqrt ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 = d23 ∧ d12^2 + d23^2 = d31^2

/-- The main theorem -/
theorem hyperbola_eccentricity_squared (h : Hyperbola) (l : Line) (A B : Point) :
  let (F1, F2) := foci h
  l.p1 = F1 →
  on_hyperbola h A →
  on_hyperbola h B →
  is_isosceles_right_triangle A B F2 →
  (h.a^2 + h.b^2) / h.a^2 = 4 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_squared_l1116_111688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_journey_time_l1116_111681

/-- Represents the journey from home to school -/
structure Journey where
  walkSpeed : ℝ
  runSpeed : ℝ
  totalDistance : ℝ
  walkDistance : ℝ
  runDistance : ℝ
  totalTime : ℝ

/-- The conditions given in the problem -/
def tuesday_journey (v s : ℝ) : Journey where
  walkSpeed := v
  runSpeed := 2 * v
  totalDistance := 3 * s
  walkDistance := 2 * s
  runDistance := s
  totalTime := 30

/-- The theorem to be proved -/
theorem wednesday_journey_time (v s : ℝ) (h1 : v > 0) (h2 : s > 0) :
  let tuesday := tuesday_journey v s
  let wednesday : Journey := {
    walkSpeed := v,
    runSpeed := 2 * v,
    totalDistance := 3 * s,
    walkDistance := s,
    runDistance := 2 * s,
    totalTime := s / v + (2 * s) / (2 * v)
  }
  tuesday.totalTime = 30 → wednesday.totalTime = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_journey_time_l1116_111681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_problem_l1116_111662

theorem salary_problem :
  ∃ (a b c d : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a + b + c + d = 4000 ∧
    0.05 * a + 0.15 * b = 0.2 * c ∧
    0.25 * d = 2 * 0.15 * b ∧
    b = 3 * c ∧
    (abs (a - 2365.55) < 0.01) ∧ 
    (abs (b - 645.15) < 0.01) ∧ 
    (abs (c - 215.05) < 0.01) ∧ 
    (abs (d - 774.18) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_problem_l1116_111662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_capacity_ratio_l1116_111693

-- Define the capacities of buckets and drum
variable (P Q D : ℚ)

-- Define the conditions
def condition1 (P Q D : ℚ) : Prop := 80 * P = D
def condition2 (P Q D : ℚ) : Prop := 60 * (P + Q) = D

-- Theorem to prove
theorem bucket_capacity_ratio 
  (h1 : condition1 P Q D) 
  (h2 : condition2 P Q D) : 
  P / Q = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_capacity_ratio_l1116_111693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_65_60_25_l1116_111690

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 65, 60, and 25 is 750 -/
theorem triangle_area_65_60_25 :
  triangleArea 65 60 25 = 750 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval triangleArea 65 60 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_65_60_25_l1116_111690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_symmetry_l1116_111633

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  (- p.b / (2 * p.a), - p.b^2 / (4 * p.a) + p.c)

/-- Checks if a point lies on a parabola -/
def lies_on (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

theorem parabola_vertex_symmetry 
  (a b : ℝ) (c d : ℝ) (ha : a ≠ 0) :
  let M := Parabola.mk a b c
  let N := Parabola.mk (-a) b d
  let P := vertex M
  let Q := vertex N
  (lies_on N P.1 P.2 ↔ lies_on M Q.1 Q.2) ∧
  (¬ lies_on N P.1 P.2 ↔ ¬ lies_on M Q.1 Q.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_symmetry_l1116_111633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_4x_minus_2cos_3x_sin_x_l1116_111685

theorem sin_4x_minus_2cos_3x_sin_x (x : ℝ) :
  Real.sin (x + π/4) = 1/3 →
  Real.sin (4*x) - 2 * Real.cos (3*x) * Real.sin x = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_4x_minus_2cos_3x_sin_x_l1116_111685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vartan_recreation_spending_l1116_111606

/-- The percentage of wages Vartan spent on recreation last week -/
def last_week_recreation_percentage : ℝ → ℝ := sorry

theorem vartan_recreation_spending (W : ℝ) (hW : W > 0) :
  let this_week_wages := 0.8 * W
  let this_week_recreation := 0.4 * this_week_wages
  this_week_recreation = 1.6 * (last_week_recreation_percentage W * W) →
  last_week_recreation_percentage W = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vartan_recreation_spending_l1116_111606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gf_has_four_zeros_l1116_111630

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + a + 1/3

/-- The function g(x) -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x^3 - 2 * b * x^2 + b * x - 4/27

/-- The composite function g(f(x)) -/
noncomputable def gf (a b : ℝ) (x : ℝ) : ℝ := g b (f a x)

/-- Theorem stating that g(f(x)) has exactly 4 zeros -/
theorem gf_has_four_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, gf a b x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gf_has_four_zeros_l1116_111630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1116_111671

/-- Represents the time (in hours) it takes Taimour to paint the fence alone -/
def taimour_time : ℝ → Prop := λ t => t > 0

/-- Represents the time (in hours) it takes Jamshid to paint the fence alone -/
def jamshid_time (t : ℝ) : ℝ := 0.5 * t

/-- Represents the time (in hours) it takes Taimour and Jamshid to paint the fence together -/
def combined_time : ℝ := 6

/-- States that the combined work rate of Taimour and Jamshid is equal to the sum of their individual work rates -/
axiom work_rate_sum (t : ℝ) : (1 / t) + (1 / (jamshid_time t)) = 1 / combined_time

theorem taimour_paint_time : taimour_time 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1116_111671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_11_piece_difference_l1116_111600

/-- Represents an L-shaped piece --/
structure LPiece where
  cells : ℕ
  is_odd : Odd cells

/-- Represents a square grid --/
structure Square where
  side : ℕ

/-- Represents a partition of a square into L-shaped pieces --/
structure Partition where
  square : Square
  pieces : List LPiece
  covers_square : (pieces.map LPiece.cells).sum = square.side * square.side

def square_120 : Square := ⟨120⟩

/-- The main theorem stating that it's impossible to have two partitions of a 120x120 square
    where the number of pieces differs by exactly 11 --/
theorem no_11_piece_difference :
  ¬∃ (p1 p2 : Partition), p1.square = square_120 ∧ p2.square = square_120 ∧
  p2.pieces.length = p1.pieces.length + 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_11_piece_difference_l1116_111600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_simplification_l1116_111631

noncomputable def q (a b c x : ℝ) : ℝ :=
  ((x + a)^4 - 3*x) / ((a - b)*(a - c)) +
  ((x + b)^4 - 3*x) / ((b - a)*(b - c)) +
  ((x + c)^4 - 3*x) / ((c - a)*(c - b))

theorem q_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  q a b c x = a^2 + b^2 + c^2 + 4*x^2 - 4*(a + b + c)*x + 12*x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_simplification_l1116_111631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_correct_l1116_111611

/-- Parameters of an ellipse given its foci and a point on the curve -/
noncomputable def ellipse_parameters (f1 f2 p : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let a := (Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) + Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)) / 2
  let h := (f1.1 + f2.1) / 2
  let k := (f1.2 + f2.2) / 2
  let c := Real.sqrt ((f2.1 - f1.1)^2 + (f2.2 - f1.2)^2) / 2
  let b := Real.sqrt (a^2 - c^2)
  (a, b, h, k)

theorem ellipse_parameters_correct :
  let f1 : ℝ × ℝ := (1, 1)
  let f2 : ℝ × ℝ := (1, 7)
  let p : ℝ × ℝ := (12, -4)
  let (a, b, h, k) := ellipse_parameters f1 f2 p
  a = (Real.sqrt 146 + Real.sqrt 242) / 2 ∧
  b = Real.sqrt (((Real.sqrt 146 + Real.sqrt 242) / 2)^2 - 9) ∧
  h = 1 ∧
  k = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_correct_l1116_111611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_eccentricity_l1116_111658

noncomputable section

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (b : ℝ) (x y : ℝ) : Prop :=
  y = b * x

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

theorem hyperbola_asymptote_eccentricity :
  ∀ b : ℝ, b > 0 →
  (∃ x y : ℝ, hyperbola b x y ∧ asymptote b 1 2) →
  b = 2 ∧ eccentricity 1 b = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_eccentricity_l1116_111658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1116_111610

/-- The rational function under consideration -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 4*x - 8) / (x^2 - 4*x + 3)

/-- The slant asymptote of f, if it exists -/
def slant_asymptote (f : ℝ → ℝ) : Option (ℝ → ℝ) := sorry

/-- Theorem stating that if the slant asymptote of f is mx + b, then m + b = 3 -/
theorem slant_asymptote_sum (m b : ℝ) : 
  slant_asymptote f = some (λ x => m * x + b) → m + b = 3 := by
  sorry

#check slant_asymptote_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1116_111610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_thirty_degrees_l1116_111684

theorem tangent_thirty_degrees (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) →  -- Point A is different from the origin
  (∃ r : ℝ, r > 0 ∧ x = r * (Real.sqrt 3 / 2) ∧ y = r * (1 / 2)) →  -- Point A is on the terminal side of a 30° angle
  y / x = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_thirty_degrees_l1116_111684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_problem_l1116_111660

/-- Given a boat traveling upstream at 4 km/h with an average round-trip speed of 5.090909090909091 km/h,
    prove that the downstream speed is approximately 7 km/h. -/
theorem boat_speed_problem (upstream_speed : ℝ) (avg_speed : ℝ) (downstream_speed : ℝ) :
  upstream_speed = 4 →
  avg_speed = 5.090909090909091 →
  avg_speed = (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed) →
  abs (downstream_speed - 7) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_problem_l1116_111660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_2_from_origin_line_through_P_max_distance_from_origin_max_distance_is_sqrt_5_l1116_111669

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between a point and a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a line passes through a point -/
def linePassesThroughPoint (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Given point P -/
def P : Point := ⟨2, -1⟩

theorem line_through_P_distance_2_from_origin :
  ∃ l : Line, linePassesThroughPoint l P ∧ 
  (l = ⟨1, 0, -2⟩ ∨ l = ⟨3, -4, -10⟩) ∧
  distancePointToLine ⟨0, 0⟩ l = 2 := by
  sorry

theorem line_through_P_max_distance_from_origin :
  ∃ l : Line, linePassesThroughPoint l P ∧
  l = ⟨2, 1, -3⟩ ∧
  ∀ l' : Line, linePassesThroughPoint l' P → 
    distancePointToLine ⟨0, 0⟩ l' ≤ distancePointToLine ⟨0, 0⟩ l := by
  sorry

theorem max_distance_is_sqrt_5 :
  let l : Line := ⟨2, 1, -3⟩
  distancePointToLine ⟨0, 0⟩ l = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_2_from_origin_line_through_P_max_distance_from_origin_max_distance_is_sqrt_5_l1116_111669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1116_111628

/-- The differential equation solution -/
noncomputable def solution (x : ℝ) : ℝ := Real.sqrt (1 + 2 * Real.log ((1 + Real.exp x) / 2))

/-- The differential equation -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (1 + Real.exp x) * y x * (deriv y x) = Real.exp x

theorem solution_satisfies_equation :
  differential_equation solution ∧ solution 0 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1116_111628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l1116_111672

/-- The distance between Alice and Bob in miles -/
def distance_AB : ℝ := 12

/-- The angle of elevation from Alice's position in radians -/
noncomputable def angle_Alice : ℝ := Real.pi / 4  -- 45 degrees in radians

/-- The angle of elevation from Bob's position in radians -/
noncomputable def angle_Bob : ℝ := Real.pi / 6  -- 30 degrees in radians

/-- The altitude of the airplane in miles -/
noncomputable def altitude : ℝ := 6 * Real.sqrt 2

theorem airplane_altitude :
  let AB := distance_AB
  let α := angle_Alice
  let β := angle_Bob
  AB * (Real.tan α) / Real.sqrt 2 = altitude := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l1116_111672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1116_111642

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  b * Real.sin B + c * Real.sin C - Real.sqrt 2 * b * Real.sin C = a * Real.sin A →
  A = π / 4 := by
  sorry

#check triangle_angle_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1116_111642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1116_111663

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (2^2 / a^2) + (3 / b^2) = 1  -- Point P(2, √3) is on the ellipse
  h4 : (a^2 - b^2) / a^2 = 3/4      -- Eccentricity is √3/2

/-- The equation of the ellipse and the abscissa of point G -/
def ellipse_properties (e : Ellipse) : Prop :=
  e.a^2 = 16 ∧ e.b^2 = 4 ∧ 
  ∃ (G : ℝ), G = 8 ∧
    ∀ (k : ℝ), 
      let x1 := (16 * k^2 + 16 * Real.sqrt (1 + 4 * k^2)) / (2 * (1 + 4 * k^2))
      let x2 := (16 * k^2 - 16 * Real.sqrt (1 + 4 * k^2)) / (2 * (1 + 4 * k^2))
      2 * x1 * x2 - 10 * (x1 + x2) + 32 = 0

/-- The main theorem stating the properties of the ellipse -/
theorem ellipse_theorem (e : Ellipse) : ellipse_properties e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1116_111663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_l1116_111616

theorem cos_double_angle_with_tan (x : ℝ) (h : Real.tan x = 2) : Real.cos (2 * x) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_l1116_111616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l1116_111623

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom parallel_lines : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c

axiom not_parallel_to_axes : ∃ (a : ℝ), a ≠ 0 ∧ ∀ x, HasDerivAt f a x ∧ HasDerivAt g a x

axiom min_value_f : ∃ x₀, ∀ x, (f x)^2 - 3 * g x ≥ (f x₀)^2 - 3 * g x₀ ∧ (f x₀)^2 - 3 * g x₀ = 11/2

theorem min_value_g : ∃ x₁, ∀ x, (g x)^2 - 3 * f x ≥ (g x₁)^2 - 3 * f x₁ ∧ (g x₁)^2 - 3 * f x₁ = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l1116_111623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_theorem_l1116_111653

/-- The volume of a prism with an equilateral triangle base -/
noncomputable def prism_volume (a b α β : ℝ) : ℝ :=
  (a^2 * b / 4) * Real.sqrt (3 - 4 * (Real.cos α^2 - Real.cos α * Real.cos β + Real.cos β^2))

/-- Represents the actual volume of the prism based on its geometric properties -/
noncomputable def volume_of_prism (a b α β : ℝ) : ℝ :=
  sorry

/-- Theorem: The volume of a prism with an equilateral triangle base of side length a,
    lateral edge b, and angles α and β between the lateral edge and the base sides -/
theorem prism_volume_theorem (a b α β : ℝ) (h_a : a > 0) (h_b : b > 0) :
  volume_of_prism a b α β = prism_volume a b α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_theorem_l1116_111653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_strip_length_l1116_111614

/-- The length of a spiral strip on a right circular cylinder -/
theorem spiral_strip_length (base_circumference height : ℝ) (h_base : base_circumference = 16) 
    (h_height : height = 8) : 
  Real.sqrt ((2 * base_circumference)^2 + height^2) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_strip_length_l1116_111614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luminosity_ratio_sun_sirius_l1116_111666

/-- The relation between magnitudes and luminosities of celestial bodies -/
def magnitude_luminosity_relation (m₁ m₂ E₁ E₂ : ℝ) : Prop :=
  m₂ - m₁ = (5/2) * Real.log (E₁/E₂) / Real.log 10

/-- The magnitude of the sun -/
def sun_magnitude : ℝ := -26.7

/-- The magnitude of Sirius -/
def sirius_magnitude : ℝ := -1.45

/-- The theorem stating the ratio of luminosities -/
theorem luminosity_ratio_sun_sirius :
  ∃ (E₁ E₂ : ℝ), 
    magnitude_luminosity_relation sun_magnitude sirius_magnitude E₁ E₂ ∧ 
    E₁ / E₂ = (10 : ℝ) ^ (10.1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luminosity_ratio_sun_sirius_l1116_111666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1116_111635

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^3 + (1-a)*x^2 - a*(a+2)*x + b

/-- The derivative of f(x) with respect to x -/
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 2*(1-a)*x - a*(a+2)

theorem problem_solution :
  ∀ a b : ℝ,
  (f a b 0 = 0 ∧ f_derivative a 0 = -3 → b = 0 ∧ (a = -3 ∨ a = 1)) ∧
  (¬ StrictMono (fun x => f a b x) ∧ ¬ StrictAnti (fun x => f a b x) →
    a ∈ Set.Ioo (-5) (-1/2) ∪ Set.Ioo (-1/2) 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1116_111635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1116_111676

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (0, 5) to the line y = 2x is √5 -/
theorem distance_to_line : distance_point_to_line 0 5 (-2) 1 0 = Real.sqrt 5 := by
  -- Unfold the definition of distance_point_to_line
  unfold distance_point_to_line
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1116_111676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_modified_chessboard_l1116_111655

/-- A chessboard is an 8x8 grid of squares. -/
def Chessboard : Type := Fin 8 × Fin 8

/-- A domino covers exactly two adjacent squares on the chessboard. -/
def Domino : Type := Chessboard × Chessboard

/-- N is the number of ways to tile a standard chessboard with 32 dominos. -/
def N : ℕ := sorry

/-- A tiling of a chessboard is a collection of dominos that cover the board without overlap. -/
def Tiling (board : Set Chessboard) : Type := List Domino

/-- Predicate to check if a tiling is valid for a given board. -/
def is_valid_tiling (board : Set Chessboard) (tiling : Tiling board) : Prop := sorry

/-- The modified chessboard with two opposite corners removed. -/
def ModifiedChessboard : Set Chessboard := sorry

/-- Theorem stating that it's impossible to tile the modified chessboard with 31 dominos. -/
theorem impossible_tiling_modified_chessboard :
  ¬∃ (tiling : Tiling ModifiedChessboard), 
    is_valid_tiling ModifiedChessboard tiling ∧ tiling.length = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_modified_chessboard_l1116_111655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_theorem_l1116_111674

-- Define the circle
def my_circle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define a point inside the circle
def inside_circle (a b r : ℝ) : Prop := a^2 + b^2 < r^2

-- Define the line l
def line_l (a b r : ℝ) : Set (ℝ × ℝ) := {p | a * p.1 + b * p.2 = r^2}

-- Define the line m (perpendicular to OM)
def line_m (a b : ℝ) : Set (ℝ × ℝ) := {p | b * p.1 - a * p.2 = 0}

-- Define parallel lines
def parallel_lines (l1 l2 : Set (ℝ × ℝ)) : Prop := 
  ∃ (k : ℝ) (c : ℝ), ∀ p, p ∈ l1 ↔ (k * p.1 + p.2 = c)

-- Define a line separate from a circle
def line_separate_circle (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ p, p ∈ l → p ∉ c

theorem chord_midpoint_theorem (a b r : ℝ) (hab : a * b ≠ 0) 
  (h_inside : inside_circle a b r) :
  parallel_lines (line_m a b) (line_l a b r) ∧ 
  line_separate_circle (line_l a b r) (my_circle r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_theorem_l1116_111674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l1116_111632

-- Define the function f(x) as noncomputable due to Real.log
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - m*x + 1)

-- State the theorem
theorem domain_condition (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m x = y) ↔ -2 < m ∧ m < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l1116_111632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_white_marbles_l1116_111670

/-- Represents the total number of marbles Peter has --/
def total_marbles : ℕ := sorry

/-- The number of orange marbles Peter has --/
def orange_marbles : ℕ := total_marbles / 2

/-- The number of purple marbles Peter has --/
def purple_marbles : ℕ := total_marbles / 5

/-- The number of silver marbles Peter has --/
def silver_marbles : ℕ := 8

/-- The number of white marbles Peter has --/
def white_marbles : ℕ := total_marbles - (orange_marbles + purple_marbles + silver_marbles)

/-- Theorem stating that the smallest possible number of white marbles is 1 --/
theorem smallest_white_marbles : 
  (∃ n : ℕ, total_marbles = n ∧ 
            orange_marbles = n / 2 ∧ 
            purple_marbles = n / 5 ∧ 
            silver_marbles = 8 ∧ 
            white_marbles > 0) →
  (∀ m : ℕ, m > 0 → white_marbles ≥ m) →
  white_marbles = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_white_marbles_l1116_111670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1116_111695

theorem power_equality (q : ℝ) : (81 : ℝ)^6 = (9 : ℝ)^q → q = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1116_111695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_equality_l1116_111651

noncomputable section

-- Define the ellipse
def W (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the left focus
def F₁ : ℝ × ℝ := (-1/2, 0)

theorem ellipse_intersection_equality 
  (A B C D : ℝ × ℝ)
  (hA : W A.1 A.2)
  (hB : W B.1 B.2)
  (hC : W C.1 C.2)
  (hD : W D.1 D.2)
  (hA_coord : A = (0, 1))
  (hB_neq_A : B ≠ A)
  (hC_neq_A : C ≠ A)
  (hC_neq_B : C ≠ B)
  (hD_neq_A : D ≠ A)
  (hD_neq_B : D ≠ B)
  (hD_neq_neg : D ≠ (0, -1))
  (E : ℝ × ℝ)
  (G : ℝ × ℝ)
  (hE : ∃ t : ℝ, E = (F₁.1, t) ∧ ∃ s : ℝ, E = (1 - s) • A + s • D)
  (hG : ∃ t : ℝ, G = (F₁.1, t) ∧ ∃ s : ℝ, G = (1 - s) • B + s • C) :
  dist E F₁ = dist F₁ G :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_equality_l1116_111651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l1116_111687

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the condition for point M
def is_on_locus (M : ℝ × ℝ) : Prop :=
  distance M A + distance M B = 2

-- Define the line segment AB
def line_segment_AB (t : ℝ) : ℝ × ℝ :=
  (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)

-- The theorem to prove
theorem locus_is_line_segment :
  ∀ M : ℝ × ℝ, is_on_locus M ↔ ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = line_segment_AB t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l1116_111687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_problem_l1116_111615

theorem binomial_probability_problem (p : ℝ) (ξ η : ℕ → ℝ) :
  (∀ k, ξ k = (Nat.choose 2 k : ℝ) * p^k * (1-p)^(2-k)) →
  (∀ k, η k = (Nat.choose 4 k : ℝ) * p^k * (1-p)^(4-k)) →
  (1 - ξ 0 = 5/9) →
  (η 2 + η 3 + η 4 = 11/27) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_problem_l1116_111615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_problem_l1116_111626

-- Define the cost and price of shirts
def cost_A (m : ℝ) := m
def cost_B (m : ℝ) := m - 10
def price_A : ℝ := 260
def price_B : ℝ := 180

-- Define the total number of shirts and the maximum number of A shirts
def total_shirts : ℕ := 300
def max_A_shirts : ℕ := 110

-- Define the discount range for A shirts
def discount_range (a : ℝ) : Prop := 60 < a ∧ a < 80

-- Define the cost equation
def cost_equation (m : ℝ) : Prop := 3 * cost_A m + 2 * cost_B m = 480

-- Define the profit function
def profit (x : ℕ) (a : ℝ) : ℝ :=
  (price_A - a - cost_A 100) * (x : ℝ) + (price_B - cost_B 100) * ((total_shirts - x) : ℝ)

-- Define the minimum profit condition
def min_profit : ℝ := 34000

-- Theorem statement
theorem shirt_problem :
  ∃ (m : ℝ),
    cost_equation m ∧
    cost_A m = 100 ∧
    cost_B m = 90 ∧
    (∃ (plans : ℕ),
      plans = 11 ∧
      ∀ x : ℕ, x ≤ max_A_shirts →
        (profit x 0 ≥ min_profit ↔ 100 ≤ x ∧ x ≤ 110)) ∧
    ∀ a : ℝ, discount_range a →
      (∀ x : ℕ, x ≤ max_A_shirts →
        (60 < a ∧ a < 70 → profit x a ≤ profit 110 a) ∧
        (a = 70 → profit x a = profit 110 a) ∧
        (70 < a ∧ a < 80 → profit x a ≤ profit 100 a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_problem_l1116_111626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operations_result_l1116_111602

-- Define the custom operations
def oplus (x y : ℝ) : ℝ := x * y - 2 * y^2

noncomputable def odot (x y : ℝ) : ℝ := Real.sqrt x + y - x * y^2

-- Theorem statement
theorem custom_operations_result :
  let x : ℝ := 9
  let y : ℝ := 3
  (oplus x y) / (odot x y) = -(3 / 25) := by
  -- Unfold the definitions
  unfold oplus odot
  -- Simplify the expressions
  simp [Real.sqrt_sq]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operations_result_l1116_111602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1116_111639

noncomputable section

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points D, E, F on the sides of the triangle
noncomputable def D : EuclideanSpace ℝ (Fin 2) := (3/4 : ℝ) • A + (1/4 : ℝ) • C
noncomputable def E : EuclideanSpace ℝ (Fin 2) := (3/4 : ℝ) • B + (1/4 : ℝ) • A
noncomputable def F : EuclideanSpace ℝ (Fin 2) := (3/4 : ℝ) • C + (1/4 : ℝ) • B

-- Define points N₁, N₂, N₃
noncomputable def N₁ (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := 
  (4/9 : ℝ) • A + (4/9 : ℝ) • C + (1/9 : ℝ) • D A C
noncomputable def N₂ (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := 
  (4/9 : ℝ) • B + (4/9 : ℝ) • A + (1/9 : ℝ) • E B A
noncomputable def N₃ (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := 
  (4/9 : ℝ) • C + (4/9 : ℝ) • B + (1/9 : ℝ) • F C B

-- Define the areas of triangles
noncomputable def area_ABC (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry
noncomputable def area_N₁N₂N₃ (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem area_ratio (A B C : EuclideanSpace ℝ (Fin 2)) :
  area_N₁N₂N₃ A B C = (1/9 : ℝ) * area_ABC A B C := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1116_111639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1116_111603

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x + abs (Real.sin x)

-- Theorem statement
theorem f_properties :
  (∃! (z₁ z₂ z₃ : ℝ), z₁ ∈ Set.Icc (-Real.pi) Real.pi ∧ 
                       z₂ ∈ Set.Icc (-Real.pi) Real.pi ∧ 
                       z₃ ∈ Set.Icc (-Real.pi) Real.pi ∧
    z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
    f z₁ = 1 ∧ f z₂ = 1 ∧ f z₃ = 1) ∧
  (∀ x : ℝ, f (2 * Real.pi - x) = f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1116_111603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_decrease_is_one_l1116_111665

/-- Proves that the decrease in average expenditure per head is 1 rupee -/
def hostel_expenditure_decrease (initial_students : ℕ) (new_students : ℕ) 
  (original_expenditure : ℕ) (expense_increase : ℕ) : ℚ :=
  let final_students := initial_students + new_students
  let final_expenditure := original_expenditure + expense_increase
  let original_average : ℚ := original_expenditure / initial_students
  let final_average : ℚ := final_expenditure / final_students
  original_average - final_average

/-- The specific problem instance -/
def problem_instance : ℚ :=
  hostel_expenditure_decrease 35 7 630 84

/-- Theorem stating that the decrease in average expenditure is 1 rupee -/
theorem expenditure_decrease_is_one : 
  problem_instance = 1 := by sorry

#eval problem_instance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_decrease_is_one_l1116_111665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_1155_l1116_111618

def sequenceList : List Nat := [3, 8, 21, 44, 85, 152, 251, 396, 593, 844]

theorem eleventh_term_is_1155 : 
  sequenceList.length = 10 → 
  (∃ next : Nat, sequenceList ++ [next] = [3, 8, 21, 44, 85, 152, 251, 396, 593, 844, 1155]) :=
by
  intro h
  use 1155
  simp [sequenceList, h]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_1155_l1116_111618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_ways_13_selection_ways_theorem_l1116_111617

/-- The number of ways to select n cards from 4 sequences of length n,
    where each index 1 to n is represented exactly once and
    no two consecutive indices (including n and 1) are from the same sequence -/
def selection_ways (n : ℕ) : ℤ :=
  3^n + 3 * (-1 : ℤ)^n

/-- Theorem stating that the number of selection ways for 13 cards is 1594320 -/
theorem selection_ways_13 :
  selection_ways 13 = 1594320 := by
  sorry

/-- Proof of the main theorem -/
theorem selection_ways_theorem (n : ℕ) (hn : n ≥ 3) :
  selection_ways n = 3^n + 3 * (-1 : ℤ)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_ways_13_selection_ways_theorem_l1116_111617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_l1116_111691

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → f x * f y + f (2008 / x) * f (2008 / y) = 2 * f (x * y)

/-- The main theorem stating that any function satisfying the functional equation
    and the given condition is constant and equal to 1 -/
theorem constant_function {f : ℝ → ℝ} (h_eq : FunctionalEq f) (h_2008 : f 2008 = 1) :
    ∀ x : ℝ, x > 0 → f x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_l1116_111691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axisymmetry_of_functions_l1116_111619

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := 1 / x
noncomputable def f2 (x : ℝ) : ℝ := Real.cos x
noncomputable def f3 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f4 (x : ℝ) : ℝ := Real.log (abs x) / Real.log 10

-- Define axisymmetry
def is_axisymmetric (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), ∀ (x : ℝ), f (a + x) = f (a - x)

-- State the theorem
theorem axisymmetry_of_functions :
  (is_axisymmetric f1) ∧
  (is_axisymmetric (fun x => f2 x) ∨ ∃ (a b : ℝ), ∀ x ∈ Set.Icc a b, is_axisymmetric (fun x => f2 x)) ∧
  (¬ is_axisymmetric f3) ∧
  (is_axisymmetric f4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axisymmetry_of_functions_l1116_111619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_value_of_z_l1116_111643

/-- The absolute value of the complex number z = (1+3i)/(1-i) is equal to √5 -/
theorem abs_value_of_z : Complex.abs ((1 + 3*Complex.I) / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_value_of_z_l1116_111643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_partition_l1116_111637

theorem card_partition (n k : ℕ) (cards : Multiset ℕ) : 
  (∀ x ∈ cards, x ≤ n) → 
  (Multiset.sum cards = n.factorial * k) →
  ∃ partition : Fin k → Multiset ℕ, 
    (∀ i, Multiset.sum (partition i) = n.factorial) ∧
    (Multiset.sum (Finset.univ.sum partition) = Multiset.sum cards) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_partition_l1116_111637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_is_closed_interval_l1116_111629

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 ∧ x ≤ 2 then 2^x - 1
  else if x < 0 ∧ x ≥ -2 then -(2^(-x) - 1)
  else 0

def g (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

-- State the theorem
theorem range_of_m_is_closed_interval :
  ∀ m : ℝ, 
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, g m x₂ = f x₁) ↔ 
  m ∈ Set.Icc (-5 : ℝ) (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_is_closed_interval_l1116_111629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_special_matrix_l1116_111686

open Matrix BigOperators

theorem det_special_matrix (n : ℕ) (k : ℝ) (a : Fin n → ℝ) :
  let A : Matrix (Fin n) (Fin n) ℝ :=
    λ i j ↦ if i = j then a i ^ 2 + k else a i * a j
  det A = k^(n-1) * (k + ∑ i, (a i)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_special_matrix_l1116_111686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_fee_is_2_35_l1116_111682

/-- Represents the taxi service pricing model -/
structure TaxiService where
  charge_per_segment : ℚ
  segment_length : ℚ
  trip_distance : ℚ
  total_charge : ℚ

/-- Calculates the initial fee for a taxi trip -/
noncomputable def initial_fee (ts : TaxiService) : ℚ :=
  ts.total_charge - (ts.charge_per_segment * (ts.trip_distance / ts.segment_length))

/-- Theorem: The initial fee for Jim's taxi service is $2.35 -/
theorem initial_fee_is_2_35 (ts : TaxiService) 
  (h1 : ts.charge_per_segment = 35/100)
  (h2 : ts.segment_length = 2/5)
  (h3 : ts.trip_distance = 36/10)
  (h4 : ts.total_charge = 11/2) : 
  initial_fee ts = 47/20 := by
  sorry

#eval (47:ℚ)/20 -- To show that 47/20 is indeed equal to 2.35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_fee_is_2_35_l1116_111682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_strategy_l1116_111668

/-- Represents a cell on the 8x8 grid --/
structure Cell where
  row : Fin 8
  col : Fin 8
deriving DecidableEq

/-- Represents the color of a cell --/
inductive Color
  | White
  | Black
deriving DecidableEq

/-- Represents the state of the game board --/
def Board := Cell → Color

/-- Checks if two cells are adjacent --/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col.val + 1 = c2.col.val ∨ c2.col.val + 1 = c1.col.val)) ∨
  (c1.col = c2.col ∧ (c1.row.val + 1 = c2.row.val ∨ c2.row.val + 1 = c1.row.val))

/-- Checks if a cell is a corner of a 5x5 square --/
def isCornerOf5x5 (c : Cell) : Prop :=
  (c.row.val % 4 = 0 ∨ c.row.val % 4 = 4) ∧
  (c.col.val % 4 = 0 ∨ c.col.val % 4 = 4)

/-- Represents a valid move by Player A --/
structure MoveA where
  cell1 : Cell
  cell2 : Cell
  adj : adjacent cell1 cell2

/-- Represents a valid move by Player B --/
structure MoveB where
  cell : Cell

/-- The main theorem stating Player B's winning strategies --/
theorem player_b_strategy (initialBoard : Board) :
  (∀ (b : Board), ∃ (move : MoveB), 
    ∀ (c : Cell), isCornerOf5x5 c → (Function.update b move.cell Color.White) c = Color.White) ∧
  (∃ (b : Board), ∀ (move : MoveB), 
    ∃ (c1 c2 : Cell), isCornerOf5x5 c1 ∧ isCornerOf5x5 c2 ∧ 
    (Function.update b move.cell Color.White) c1 = Color.Black ∧ 
    (Function.update b move.cell Color.White) c2 = Color.Black) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_strategy_l1116_111668
