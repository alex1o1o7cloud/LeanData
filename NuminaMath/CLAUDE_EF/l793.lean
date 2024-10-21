import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l793_79308

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi/2) * Real.cos (3*Real.pi/2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

theorem problem_solution (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) :
  (f α = -Real.cos α) ∧ (f (2*α) = -23/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l793_79308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_half_alpha_l793_79377

theorem cos_pi_minus_half_alpha (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : Real.sin α = 3/5) :
  Real.cos (π - α/2) = -3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_half_alpha_l793_79377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_not_odd_l793_79316

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.sin x + Real.cos x)

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧  -- Domain is ℝ
  (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧  -- Periodic with period 2π
  (∃ m M : ℝ, ∀ x : ℝ, m ≤ f x ∧ f x ≤ M)  -- Has both maximum and minimum values
:= by
  sorry  -- Skip the proof for now

-- Additional theorem to show f is not an odd function
theorem f_not_odd : ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry  -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_not_odd_l793_79316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_max_value_l793_79314

open Real BigOperators

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

theorem perpendicular_vectors_max_value 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h_n : n > 0) 
  (h_perp : 2 * (sequence_sum a n) = (n + 1) * (n + 1)) :
  ∃ (m : ℕ), m > 0 ∧ 
    (∀ (k : ℕ), k > 0 → a k / (a (k + 1) * a (k + 4)) ≤ 1 / 9) ∧
    a m / (a (m + 1) * a (m + 4)) = 1 / 9 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_max_value_l793_79314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_portrait_location_l793_79334

theorem portrait_location :
  -- Define the propositions
  let p : Prop := true  -- Representing "The portrait is in the gold box"
  let q : Prop := true  -- Representing "The portrait is not in the silver box"
  let r : Prop := true  -- Representing "The portrait is not in the gold box"
  -- Define the conditions
  ∀ (gold silver lead : Prop),
    (gold ↔ p) →  -- Gold box contains the portrait iff p is true
    (silver ↔ ¬q) →  -- Silver box contains the portrait iff q is false
    (lead ↔ r) →  -- Lead box contains the portrait iff r is true
    (gold ∨ silver ∨ lead) →  -- The portrait is in one of the boxes
    (¬(gold ∧ silver) ∧ ¬(gold ∧ lead) ∧ ¬(silver ∧ lead)) →  -- The portrait is in only one box
    (p ↔ ¬r) →  -- p and r are contradictory
    ((p ∧ ¬q ∧ ¬r) ∨ (¬p ∧ q ∧ ¬r) ∨ (¬p ∧ ¬q ∧ r)) →  -- Only one proposition is true
    silver  -- The portrait is in the silver box
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_portrait_location_l793_79334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_96_km_l793_79355

/-- Calculates the distance to a destination given rowing speed, current velocity, and round trip time -/
noncomputable def distance_to_destination (rowing_speed : ℝ) (current_velocity : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := rowing_speed - current_velocity
  let downstream_speed := rowing_speed + current_velocity
  (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))

/-- Theorem stating that the distance to the destination is 96 km given the problem conditions -/
theorem distance_is_96_km (rowing_speed : ℝ) (current_velocity : ℝ) (total_time : ℝ)
    (h1 : rowing_speed = 10)
    (h2 : current_velocity = 2)
    (h3 : total_time = 20) :
    distance_to_destination rowing_speed current_velocity total_time = 96 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_to_destination 10 2 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_96_km_l793_79355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_candy_consumption_l793_79368

/-- The number of pieces of candy Megan received from neighbors -/
noncomputable def candy_from_neighbors : ℝ := 11

/-- The number of pieces of candy Megan received from her older sister -/
noncomputable def candy_from_sister : ℝ := 5

/-- The number of days the candy lasted -/
noncomputable def days_candy_lasted : ℝ := 2

/-- The number of pieces of candy Megan ate per day -/
noncomputable def candy_per_day : ℝ := (candy_from_neighbors + candy_from_sister) / days_candy_lasted

theorem megan_candy_consumption :
  candy_per_day = 8 := by
  -- Unfold the definitions
  unfold candy_per_day candy_from_neighbors candy_from_sister days_candy_lasted
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_megan_candy_consumption_l793_79368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_seventh_test_score_l793_79367

def is_valid_score_set (scores : Finset ℕ) : Prop :=
  scores.card = 8 ∧
  (∀ s, s ∈ scores → 88 ≤ s ∧ s ≤ 100) ∧
  (∀ s₁ s₂, s₁ ∈ scores → s₂ ∈ scores → s₁ ≠ s₂ → s₁ ≠ s₂)

def has_integer_averages (scores : List ℕ) : Prop :=
  ∀ k, k ∈ Finset.range scores.length → (scores.take (k + 1)).sum % (k + 1) = 0

theorem lucas_seventh_test_score 
  (scores : Finset ℕ) 
  (h_valid : is_valid_score_set scores) 
  (h_avg : has_integer_averages (scores.toList))
  (h_eighth : 94 ∈ scores) :
  ∃ seventh_score, seventh_score ∈ scores ∧ seventh_score = 100 ∧ 
    ((scores.toList.filter (· ≠ 94)).indexOf 100 = 6) := by
  sorry

#check lucas_seventh_test_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_seventh_test_score_l793_79367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_with_three_count_l793_79338

/-- The count of four-digit positive integers with thousands digit 3 -/
def count_four_digit_with_three : ℕ :=
  10 * 10 * 10

theorem four_digit_with_three_count :
  count_four_digit_with_three = 1000 := by
  rfl

#eval count_four_digit_with_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_with_three_count_l793_79338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l793_79369

noncomputable section

-- Define the geometric sequence and its sum
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)
def sum_geometric (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

-- State the theorem
theorem geometric_sequence_properties
  (a₁ : ℝ) (q : ℝ) (h_arith : ∃ d : ℝ, sum_geometric a₁ q 2 - sum_geometric a₁ q 1 = sum_geometric a₁ q 3 - sum_geometric a₁ q 2)
  (h_diff : a₁ - geometric_sequence a₁ q 3 = 3) :
  q = -1/2 ∧ ∀ n : ℕ, sum_geometric a₁ q n = (8/3) * (1 - (-1/2)^n) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l793_79369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_males_in_band_only_l793_79348

/-- Represents a musical group in the school -/
inductive MusicalGroup
| Band
| Orchestra
| Choir

/-- Represents the gender of a student -/
inductive Gender
| Female
| Male

/-- The number of students in each group by gender -/
def group_count : MusicalGroup → Gender → ℕ
| MusicalGroup.Band, Gender.Female => 50
| MusicalGroup.Band, Gender.Male => 40
| MusicalGroup.Orchestra, Gender.Female => 40
| MusicalGroup.Orchestra, Gender.Male => 50
| MusicalGroup.Choir, Gender.Female => 30
| MusicalGroup.Choir, Gender.Male => 45

/-- The number of female students in two groups -/
def female_overlap : MusicalGroup → MusicalGroup → ℕ
| MusicalGroup.Band, MusicalGroup.Orchestra => 20
| MusicalGroup.Band, MusicalGroup.Choir => 15
| MusicalGroup.Orchestra, MusicalGroup.Choir => 10
| _, _ => 0

/-- The number of female students in all three groups -/
def female_all_groups : ℕ := 5

/-- The total number of students in any group or combination -/
def total_students : ℕ := 120

theorem males_in_band_only : 
  (group_count MusicalGroup.Band Gender.Male) - 
  ((group_count MusicalGroup.Band Gender.Male) - 
   ((total_students - 
     ((group_count MusicalGroup.Band Gender.Female) + 
      (group_count MusicalGroup.Orchestra Gender.Female) + 
      (group_count MusicalGroup.Choir Gender.Female) - 
      (female_overlap MusicalGroup.Band MusicalGroup.Orchestra) - 
      (female_overlap MusicalGroup.Band MusicalGroup.Choir) - 
      (female_overlap MusicalGroup.Orchestra MusicalGroup.Choir) + 
      female_all_groups)) - 
    ((group_count MusicalGroup.Orchestra Gender.Male) + 
     (group_count MusicalGroup.Choir Gender.Male)))) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_males_in_band_only_l793_79348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_zero_implies_x_value_l793_79343

open Real

theorem tan_sum_zero_implies_x_value (x : ℝ) :
  tan (4 * x) + tan (6 * x) = 0 →
  0 ≤ x →
  x < π →
  x = π / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_zero_implies_x_value_l793_79343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_min_value_f_min_at_pi_third_l793_79364

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 1/2 * (Real.cos x)^2 - 1/2 * (Real.sin x)^2 + 1 - Real.sqrt 3 * Real.sin x * Real.cos x

/-- The period of f(x) is π -/
theorem f_period : ∀ x, f (x + Real.pi) = f x := by sorry

/-- The minimum value of f(x) on [0, π/2] is 0 -/
theorem f_min_value : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 0 ∧ ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≥ f x := by sorry

/-- The minimum of f(x) on [0, π/2] occurs at x = π/3 -/
theorem f_min_at_pi_third : f (Real.pi / 3) = 0 ∧ ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ f (Real.pi / 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_min_value_f_min_at_pi_third_l793_79364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_100_sum_or_diff_l793_79325

theorem divisible_by_100_sum_or_diff (S : Finset ℤ) (h : S.card = 52) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (100 ∣ (a + b) ∨ 100 ∣ (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_100_sum_or_diff_l793_79325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_condition_l793_79301

theorem integer_pair_condition (x y : ℕ) : 
  x > 1 ∧ y > 0 → 
  (∃ k : ℕ, (Int.floor (↑x^2 / ↑y : ℚ) + 1 : ℤ) = k * x) ↔ 
  ((x = 2 ∧ y = 4) ∨ (∃ t : ℕ, t > 1 ∧ x = t ∧ y = t + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_condition_l793_79301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_for_given_properties_l793_79329

/-- An ellipse with specific properties -/
structure Ellipse where
  -- Center at the origin and major axis along y-axis are implicit in the structure
  eccentricity : ℝ
  focal_distance_sum : ℝ

/-- The equation of an ellipse given its properties -/
def ellipse_equation (x y : ℝ) : Prop :=
  y^2 / 36 + x^2 / 9 = 1

/-- Theorem stating the equation of the ellipse with given properties -/
theorem ellipse_equation_for_given_properties (e : Ellipse) 
  (h_ecc : e.eccentricity = Real.sqrt 3 / 2)
  (h_sum : e.focal_distance_sum = 12) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation p.1 p.2} ↔ 
    ∃ p : ℝ × ℝ, p ∈ {q : ℝ × ℝ | (∃ (f₁ f₂ : ℝ × ℝ), 
      dist p f₁ + dist p f₂ = e.focal_distance_sum ∧
      dist (0, 0) f₁ = dist (0, 0) f₂ ∧
      f₁.2 = -f₂.2 ∧ f₁.1 = 0)} ∧ 
    p.1 = x ∧ p.2 = y :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_for_given_properties_l793_79329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_constant_term_l793_79363

theorem min_n_for_constant_term (x : ℝ) (n : ℕ) :
  (∃ k : ℕ, k ≤ n ∧ (n.choose k) * (x^(1/2 : ℝ))^(n - k) * (3 * x^(-(1/3 : ℝ)))^k = 1) →
  n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_constant_term_l793_79363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_four_horses_meet_l793_79381

def horse_times : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_at_start (time : Nat) (lap_time : Nat) : Bool :=
  time % lap_time = 0

def count_horses_at_start (time : Nat) (times : List Nat) : Nat :=
  (times.filter (λ lap_time => is_at_start time lap_time)).length

theorem least_time_four_horses_meet :
  ∃ (T : Nat),
    T > 0 ∧
    count_horses_at_start T horse_times = 4 ∧
    ∀ (t : Nat), t > 0 ∧ t < T → count_horses_at_start t horse_times ≠ 4 :=
by sorry

#eval count_horses_at_start 210 horse_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_four_horses_meet_l793_79381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l793_79390

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 2)^2 + (y + 1)^2) = 4

noncomputable def positive_asymptote_slope : ℝ := Real.sqrt 3 / 3

/-- Theorem stating that the positive slope of an asymptote of the given hyperbola is √3/3 -/
theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ positive_asymptote_slope = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l793_79390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_value_l793_79375

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.exp (3 * x) + 1) + a * x

theorem even_function_value (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = -3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_value_l793_79375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_sum_equality_l793_79302

/-- Represents the state of coins on the table -/
structure CoinState where
  gold : ℕ
  silver : ℕ

/-- Represents the action taken and the number recorded -/
inductive CoinAction where
  | AddGold (silver_count : ℕ)
  | RemoveSilver (gold_count : ℕ)

/-- The coin game process -/
def CoinGame := List CoinAction

/-- Apply an action to a coin state -/
def applyAction (state : CoinState) (action : CoinAction) : CoinState :=
  match action with
  | CoinAction.AddGold _ => { state with gold := state.gold + 1 }
  | CoinAction.RemoveSilver _ => { state with silver := state.silver - 1 }

theorem coin_game_sum_equality 
  (initial_silver : ℕ) 
  (final_gold : ℕ) 
  (game : CoinGame) 
  (h_start : game.foldl applyAction { gold := 0, silver := initial_silver } = { gold := final_gold, silver := 0 }) :
  (game.filterMap (λ action => 
    match action with
    | CoinAction.AddGold n => some n
    | _ => none
  )).sum = 
  (game.filterMap (λ action => 
    match action with
    | CoinAction.RemoveSilver n => some n
    | _ => none
  )).sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_sum_equality_l793_79302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l793_79304

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 12) + 1

theorem max_value_of_expression (x₁ x₂ : ℝ) 
  (h1 : g x₁ * g x₂ = 9)
  (h2 : x₁ ∈ Set.Icc (-2 * Real.pi) (2 * Real.pi))
  (h3 : x₂ ∈ Set.Icc (-2 * Real.pi) (2 * Real.pi)) :
  (∀ y₁ y₂ : ℝ, g y₁ * g y₂ = 9 → 
    y₁ ∈ Set.Icc (-2 * Real.pi) (2 * Real.pi) → 
    y₂ ∈ Set.Icc (-2 * Real.pi) (2 * Real.pi) → 
    2 * y₁ - y₂ ≤ 2 * x₁ - x₂) →
  2 * x₁ - x₂ = 49 * Real.pi / 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l793_79304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l793_79315

-- Define the function f as noncomputable due to its dependency on Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x + 1)

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, x ≤ 1 → f a x ∈ Set.Icc 0 (Real.sqrt 2)) ↔ a ∈ Set.Icc (-1) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l793_79315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_period_is_pi_over_two_l793_79394

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (4 * x - Real.pi / 3)

def is_periodic (h : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, h (x + p) = h x

theorem min_positive_period_f :
  ∃ (p : ℝ), p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q → q < p → ¬ is_periodic f q :=
by sorry

theorem period_is_pi_over_two :
  ∃ (p : ℝ), p = Real.pi / 2 ∧ is_periodic f p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_period_is_pi_over_two_l793_79394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_quadratic_iff_m_eq_two_l793_79358

/-- The equation (m-2)x^2+5x+m^2-2m=0 is not a quadratic equation in x if and only if m = 2 -/
theorem not_quadratic_iff_m_eq_two (m : ℝ) : 
  (∀ x, (m - 2) * x^2 + 5 * x + m^2 - 2 * m = 0 → (m - 2 = 0)) ↔ 
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_quadratic_iff_m_eq_two_l793_79358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_distribution_theorem_l793_79337

/-- Represents a sock distribution among children -/
structure SockDistribution where
  total_socks : ℕ
  num_children : ℕ
  socks_per_child : ℕ
  h_total : total_socks = num_children * socks_per_child

/-- Predicate to check if a sock distribution satisfies the given conditions -/
def satisfies_conditions (d : SockDistribution) : Prop :=
  (d.total_socks = 9) ∧
  (∀ (subset : Finset ℕ), subset.card = 4 → 
    ∃ (child : ℕ), child < d.num_children ∧ 
    (subset.filter (λ sock ↦ sock % d.num_children = child)).card ≥ 2) ∧
  (∀ (subset : Finset ℕ), subset.card = 5 → 
    ∀ (child : ℕ), child < d.num_children → 
    (subset.filter (λ sock ↦ sock % d.num_children = child)).card ≤ 3)

/-- Theorem stating the existence of a sock distribution satisfying the conditions -/
theorem sock_distribution_theorem :
  ∃ (d : SockDistribution), satisfies_conditions d ∧ d.num_children = 3 ∧ d.socks_per_child = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_distribution_theorem_l793_79337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l793_79327

/-- Given two curves f(x) = e^x and g(x) = ax^2 - a (a ≠ 0), where the tangent line to f(x) at x = 0
    is tangent to g(x), the equation of the line passing through the tangent point and perpendicular
    to the tangent line is x + y + 1 = 0. -/
theorem perpendicular_line_equation
  (f g : ℝ → ℝ)
  (hf : ∀ x, f x = Real.exp x)
  (hg : ∃ a ≠ 0, ∀ x, g x = a * x^2 - a)
  (h_tangent : ∃ x₀, (deriv f 0) * x₀ + f 0 = g x₀ ∧ deriv f 0 = deriv g x₀) :
  ∃ m b, m = -1 ∧ b = 1 ∧ ∀ x y, y = m * x + b ↔ x + y + 1 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l793_79327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l793_79383

structure Package where
  length : ℚ
  width : ℚ
  weight : ℚ

def extra_charge (p : Package) : Bool :=
  p.length / p.width < 3/2 ∨ p.length / p.width > 3 ∨ p.weight > 5

def packages : List Package := [
  ⟨8, 6, 4⟩,
  ⟨12, 4, 6⟩,
  ⟨7, 7, 5⟩,
  ⟨14, 4, 3⟩
]

theorem extra_charge_count : 
  (packages.filter extra_charge).length = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l793_79383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_l793_79312

/-- An ellipse with equation y^2/16 + x^2/9 = 1 -/
def is_ellipse (P : ℝ × ℝ) : Prop :=
  (P.2^2 / 16) + (P.1^2 / 9) = 1

/-- A hyperbola with equation y^2/4 - x^2/5 = 1 -/
def is_hyperbola (P : ℝ × ℝ) : Prop :=
  (P.2^2 / 4) - (P.1^2 / 5) = 1

/-- The distance between two points in ℝ² -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem ellipse_hyperbola_intersection (F₁ F₂ P : ℝ × ℝ) 
  (h_ellipse : is_ellipse P)
  (h_hyperbola : is_hyperbola P)
  (h_shared_foci : ∀ Q, is_ellipse Q ↔ 
    distance Q F₁ + distance Q F₂ = 2 * 4 ∧
    distance Q F₁ - distance Q F₂ = 2 * 2) :
  distance P F₁ * distance P F₂ = 12 := by
  sorry

#check ellipse_hyperbola_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_l793_79312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_equals_six_l793_79341

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define our specific triangle
noncomputable def ourTriangle : Triangle where
  A := 75 * Real.pi / 180  -- Convert to radians
  B := 45 * Real.pi / 180  -- Convert to radians
  C := Real.pi - (75 * Real.pi / 180) - (45 * Real.pi / 180)  -- Internal angle sum theorem
  a := 0  -- We don't know this value, but it's not needed for the proof
  b := 0  -- This is what we're trying to prove
  c := 3 * Real.sqrt 6

-- State the theorem
theorem side_b_equals_six (t : Triangle) (h1 : t = ourTriangle) : t.b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_equals_six_l793_79341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_price_increase_l793_79320

theorem food_price_increase 
  (student_decrease : ℝ) 
  (consumption_decrease : ℝ) 
  (student_decrease_value : student_decrease = 0.09)
  (consumption_decrease_value : consumption_decrease = 0.08424908424908429) :
  let new_student_ratio := 1 - student_decrease
  let new_consumption_ratio := 1 - consumption_decrease
  let price_increase_ratio := 1 / (new_student_ratio * new_consumption_ratio)
  abs (price_increase_ratio - 1 - 0.0989) < 0.0001 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_price_increase_l793_79320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l793_79398

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x ≤ 2) ∧
  (∀ x, f x ≥ -2) ∧
  (∀ x, f x = 2 ↔ ∃ k : ℤ, x = 2 * k * Real.pi / 3 + Real.pi / 18) ∧
  (∀ x, f x = -2 ↔ ∃ k : ℤ, x = 2 * k * Real.pi / 3 - 5 * Real.pi / 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l793_79398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_exp_l793_79345

noncomputable def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => a n + 2 * n * a (n - 1) + 9 * n * (n - 1) * a (n - 2) + 8 * n * (n - 1) * (n - 2) * a (n - 3)

noncomputable def series_sum : ℝ := ∑' n : ℕ, (10 : ℝ)^n * (a n : ℝ) / n.factorial

theorem series_sum_equals_exp :
  series_sum = Real.exp 182216.6667 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_exp_l793_79345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_concurrency_l793_79310

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (externally_tangent : Circle → Circle → Point → Prop)
variable (tangent_line : Circle → Point → Prop)
variable (intersects : Point → Point → Circle → Point → Point → Prop)
variable (extends_to : Point → Point → Circle → Point → Prop)
variable (midpoint_of_arc : Circle → Point → Point → Point → Point → Prop)
variable (intersection : Point → Point → Circle → Point → Prop)
variable (concurrent : Point → Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem circles_concurrency
  (C₁ C₂ : Circle)
  (A B C D E F H : Point)
  (h1 : externally_tangent C₁ C₂ A)
  (h2 : tangent_line C₁ B)
  (h3 : intersects B C C₂ C D)
  (h4 : extends_to A B C₂ E)
  (h5 : midpoint_of_arc C₂ C D E F)
  (h6 : intersection B F C₂ H) :
  concurrent C D A F E H :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_concurrency_l793_79310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_theorem_l793_79311

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Pyramid where
  apex : Point
  base : List Point
  altitude_foot : Point

structure Sphere where
  center : Point
  radius : ℝ

-- Define helper functions
def sphere_touches_faces (S : Sphere) (P : Pyramid) : Prop := sorry

def on_lateral_edge (A : Point) (P : Pyramid) : Prop := sorry

def segment_passes_through_point (A B C : Point) : Prop := sorry

def is_tangency_point (K : Point) (S : Sphere) (P : Pyramid) : Prop := sorry

-- Define the main theorem
theorem tangency_point_theorem 
  (P : Pyramid) 
  (S : Sphere) 
  (A B C D : Point) 
  (K L M N : Point) : 
  (S.center = P.altitude_foot) →
  sphere_touches_faces S P →
  on_lateral_edge A P →
  on_lateral_edge B P →
  on_lateral_edge C P →
  on_lateral_edge D P →
  segment_passes_through_point A B K →
  segment_passes_through_point B C L →
  segment_passes_through_point C D M →
  is_tangency_point K S P →
  is_tangency_point L S P →
  is_tangency_point M S P →
  is_tangency_point N S P →
  segment_passes_through_point A D N :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_theorem_l793_79311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_plummet_distance_l793_79397

/-- Rocket flight parameters -/
structure RocketFlight where
  ascent_time : ℝ
  ascent_speed : ℝ
  descent_time : ℝ
  average_speed : ℝ

/-- Calculate the plummet distance of a rocket flight -/
def plummet_distance (flight : RocketFlight) : ℝ :=
  let total_time := flight.ascent_time + flight.descent_time
  let total_distance := flight.average_speed * total_time
  let ascent_distance := flight.ascent_speed * flight.ascent_time
  total_distance - ascent_distance

/-- Theorem: The rocket plummets 600 meters -/
theorem rocket_plummet_distance :
  let flight : RocketFlight := {
    ascent_time := 12,
    ascent_speed := 150,
    descent_time := 3,
    average_speed := 160
  }
  plummet_distance flight = 600 := by
  sorry

#eval plummet_distance {
  ascent_time := 12,
  ascent_speed := 150,
  descent_time := 3,
  average_speed := 160
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_plummet_distance_l793_79397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problem_l793_79366

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := 
  Int.floor x

-- Define the decimal part function
noncomputable def decPart (x : ℝ) : ℝ := 
  x - (intPart x : ℝ)

-- Define variables a and b
variable (a b : ℝ)

-- State the theorem
theorem sqrt_problem (h1 : decPart (Real.sqrt 7) = a) 
                     (h2 : intPart (Real.sqrt 37) = b) : 
  Real.sqrt (a + b - Real.sqrt 7) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problem_l793_79366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_only_time_l793_79352

/-- Represents the speed of a vehicle -/
structure Speed where
  value : ℝ

/-- Represents the time spent traveling -/
structure Time where
  value : ℝ

/-- Represents a distance -/
structure Distance where
  value : ℝ

/-- The total distance traveled is the product of speed and time -/
def travel (s : Speed) (t : Time) : Distance where
  value := s.value * t.value

/-- Addition of distances -/
instance : HAdd Distance Distance Distance where
  hAdd d1 d2 := ⟨d1.value + d2.value⟩

/-- The theorem to be proved -/
theorem motorcycle_only_time 
  (vm : Speed) -- Speed of motorcycle
  (vb : Speed) -- Speed of bicycle
  (d : Distance) -- Total distance between A and B
  (h1 : travel vm ⟨12⟩ + travel vb ⟨9⟩ = d) -- First scenario
  (h2 : travel vb ⟨21⟩ + travel vm ⟨8⟩ = d) -- Second scenario
  : travel vm ⟨15⟩ = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_only_time_l793_79352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_statements_l793_79396

theorem two_correct_statements :
  ∃ (S : Finset (Prop)),
    S.card = 2 ∧
    S ⊆ {
      (∀ a b : ℝ, a - b > 0 → a > 0 ∧ b > 0),
      (∀ a b : ℝ, a - b = a + (-b)),
      (∀ a : ℝ, a - (-a) = 0),
      (∀ a : ℝ, 0 - a = -a)
    } ∧
    (∀ p ∈ S, p) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_statements_l793_79396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l793_79336

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 * x^2 + 1)

def point : ℝ × ℝ := (1, 2)

def tangent_line (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0

theorem tangent_line_at_point :
  let (x₀, y₀) := point
  (∀ x, f x = Real.sqrt (3 * x^2 + 1)) →
  f x₀ = y₀ →
  (∃ k : ℝ, ∀ x, tangent_line x (y₀ + k * (x - x₀))) :=
by sorry

#check tangent_line_at_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l793_79336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_17_l793_79372

/-- The area of the shaded region in a 4 × 5 rectangle with a circle of diameter 2 removed -/
noncomputable def shaded_area : ℝ := 20 - Real.pi

/-- The whole number closest to the shaded area -/
def closest_whole_number : ℕ := 17

theorem shaded_area_closest_to_17 : 
  ∀ n : ℕ, |shaded_area - ↑closest_whole_number| ≤ |shaded_area - ↑n| := by
  sorry

#eval closest_whole_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_17_l793_79372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shapes_l793_79384

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  -- Add other necessary properties of a cube

-- Define the possible shapes
inductive Shape
  | Rectangle
  | NonRectangleParallelogram
  | TetrahedronIsoscelesRightAndEquilateral
  | TetrahedronAllEquilateral
  | TetrahedronAllRight

-- Function to check if a shape can be formed by 4 vertices of a cube
def canFormShape (c : Cube) (s : Shape) : Prop :=
  ∃ (v : Finset (Fin 8)), v.card = 4 ∧ v ⊆ c.vertices ∧
    match s with
    | Shape.Rectangle => sorry
    | Shape.NonRectangleParallelogram => sorry
    | Shape.TetrahedronIsoscelesRightAndEquilateral => sorry
    | Shape.TetrahedronAllEquilateral => sorry
    | Shape.TetrahedronAllRight => sorry

-- Theorem stating which shapes can be formed
theorem cube_shapes (c : Cube) :
  canFormShape c Shape.Rectangle ∧
  canFormShape c Shape.TetrahedronIsoscelesRightAndEquilateral ∧
  canFormShape c Shape.TetrahedronAllEquilateral ∧
  canFormShape c Shape.TetrahedronAllRight ∧
  ¬canFormShape c Shape.NonRectangleParallelogram :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shapes_l793_79384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_two_l793_79344

/-- The number of intersection points between two circles -/
def intersection_count (c1 c2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- First circle: (x - 2)² + y² = 4 -/
def circle1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

/-- Second circle: x² + (y - 4)² = 16 -/
def circle2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 4)^2 = 16}

/-- Theorem stating that the number of intersection points between the two circles is 2 -/
theorem intersection_count_is_two :
  intersection_count circle1 circle2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_two_l793_79344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_wins_l793_79392

/-- Represents a game state with two piles of candies -/
structure GameState where
  pile1 : Nat
  pile2 : Nat

/-- Represents a player in the game -/
inductive Player
  | A
  | B

/-- Defines a valid move in the game -/
def validMove (s : GameState) (s' : GameState) : Prop :=
  (s.pile1 = 0 ∧ s'.pile2 < s.pile2) ∨
  (s.pile2 = 0 ∧ s'.pile1 < s.pile1)

/-- Defines when a player wins the game -/
def wins (p : Player) (s : GameState) : Prop :=
  s.pile1 = 0 ∧ s.pile2 = 0

/-- Defines a winning strategy for a player -/
inductive hasWinningStrategy : Player → GameState → Prop
  | base (p : Player) (s : GameState) : wins p s → hasWinningStrategy p s
  | step (p : Player) (s : GameState) (move : GameState) :
      validMove s move →
      (∀ (opponent_move : GameState), 
        validMove move opponent_move → 
        hasWinningStrategy p opponent_move) →
      hasWinningStrategy p s

/-- The main theorem: Player A has a winning strategy in the initial game state -/
theorem player_A_wins : hasWinningStrategy Player.A ⟨33, 35⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_wins_l793_79392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_isosceles_l793_79395

-- Define the points in the plane
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the condition that the points are distinct
variable (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

-- Define the given equation
variable (h_equation : ((D - B) + (D - C) - 2 • (D - A)) • ((A - B) - (A - C)) = 0)

-- Theorem statement
theorem triangle_ABC_is_isosceles : 
  ∃ (X Y : EuclideanSpace ℝ (Fin 2)), ((X = B ∧ Y = C) ∨ (X = C ∧ Y = A) ∨ (X = A ∧ Y = B)) ∧ ‖A - X‖ = ‖A - Y‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_isosceles_l793_79395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_integers_sum_thirty_one_is_max_sum_formula_max_consecutive_integers_before_500_l793_79370

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by sorry

theorem thirty_one_is_max : ∀ m : ℕ, m > 31 → m * (m + 1) > 1000 := by sorry

theorem sum_formula (n : ℕ) : 2 * (Finset.sum (Finset.range n) (λ i => i + 1)) = n * (n + 1) := by sorry

theorem max_consecutive_integers_before_500 :
  (∃ n : ℕ, n > 0 ∧ 
    (∀ m : ℕ, m > n → 2 * (Finset.sum (Finset.range m) (λ i => i + 1)) > 1000) ∧
    2 * (Finset.sum (Finset.range n) (λ i => i + 1)) ≤ 1000) ∧
  (∀ n : ℕ, 
    (∀ m : ℕ, m > n → 2 * (Finset.sum (Finset.range m) (λ i => i + 1)) > 1000) ∧
    2 * (Finset.sum (Finset.range n) (λ i => i + 1)) ≤ 1000 → n ≤ 31) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_integers_sum_thirty_one_is_max_sum_formula_max_consecutive_integers_before_500_l793_79370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_envelopes_require_fee_l793_79386

/-- Represents an envelope with length and height measurements. -/
structure Envelope where
  length : ℚ
  height : ℚ

/-- Determines if an extra fee is required for an envelope. -/
def requiresExtraFee (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 3/2 || ratio > 3

/-- The list of envelopes from the problem. -/
def envelopes : List Envelope := [
  ⟨3, 2⟩,   -- Envelope X
  ⟨10, 3⟩,  -- Envelope Y
  ⟨5, 5⟩,   -- Envelope Z
  ⟨15, 4⟩   -- Envelope W
]

/-- Theorem stating that exactly 3 envelopes require an extra fee. -/
theorem three_envelopes_require_fee : 
  (envelopes.filter requiresExtraFee).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_envelopes_require_fee_l793_79386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l793_79306

open Real

theorem angle_relation (α β : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : β ∈ Set.Ioo 0 (π/2))
  (h3 : tan α = (1 + sin β) / cos β) : 2 * α - β = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l793_79306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_jar_size_l793_79339

/-- Proves that given the conditions, the size of the third jar type is 1 gallon -/
theorem third_jar_size (total_water : ℚ) (total_jars : ℕ) (jar_types : ℕ) :
  total_water = 28 →
  total_jars = 48 →
  jar_types = 3 →
  ∃ (third_jar_size : ℚ),
    let jars_per_type : ℕ := total_jars / jar_types
    let quart_size : ℚ := 1/4
    let half_gallon_size : ℚ := 1/2
    jars_per_type * quart_size + jars_per_type * half_gallon_size + jars_per_type * third_jar_size = total_water ∧
    third_jar_size = 1 :=
by
  intros h1 h2 h3
  use 1
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_jar_size_l793_79339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_faster_l793_79347

/-- The distance between stations A and B in kilometers -/
noncomputable def distance : ℝ := 360

/-- The speed of the train departing from station A in km/h -/
noncomputable def speed_A : ℝ := sorry

/-- The speed of the train departing from station B in km/h -/
noncomputable def speed_B : ℝ := sorry

/-- The time it takes for the train from A to reach B in hours -/
noncomputable def time_A_to_B : ℝ := distance / speed_A

/-- The time it takes for the trains to meet if A's speed was 1.5 times faster -/
noncomputable def time_to_meet : ℝ := distance / (1.5 * speed_A + speed_B)

theorem train_B_faster :
  time_A_to_B ≥ 5 ∧ time_to_meet < 2 → speed_B > speed_A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_faster_l793_79347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l793_79399

/-- The function f(x) = 1 + m/x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 + m / x

theorem problem_solution :
  -- Part 1: Prove m = 1
  (∃ m : ℝ, f m 1 = 2 ∧ m = 1) ∧
  -- Part 2: Prove f is monotonically decreasing on (0, +∞)
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f 1 x₁ > f 1 x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l793_79399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l793_79378

def coin_toss : ℕ := 3

def event_A (outcomes : Fin coin_toss → Bool) : Prop :=
  ∃ i, outcomes i = false

def event_B (outcomes : Fin coin_toss → Bool) : Prop :=
  (List.filter id (List.ofFn outcomes)).length = 1

def prob_A : ℚ := 7/8

def prob_AB : ℚ := 3/8

theorem conditional_probability_B_given_A :
  prob_AB / prob_A = 3/7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l793_79378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l793_79380

def sequenceA (n : ℕ) : ℚ := (2 * n - 1) / (2^n : ℚ)

theorem sequence_first_five_terms :
  [sequenceA 1, sequenceA 2, sequenceA 3, sequenceA 4, sequenceA 5] = [1/2, 3/4, 5/8, 7/16, 9/32] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l793_79380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_temp_range_l793_79342

/-- Represents a set of daily temperatures -/
def DailyTemperatures := List ℝ

/-- The number of days -/
def numDays : ℕ := 7

/-- Calculates the average of a list of numbers -/
noncomputable def average (temps : DailyTemperatures) : ℝ :=
  temps.sum / temps.length

/-- Finds the minimum value in a list of numbers -/
noncomputable def minTemp (temps : DailyTemperatures) : ℝ :=
  temps.minimum?.getD 0

/-- Finds the maximum value in a list of numbers -/
noncomputable def maxTemp (temps : DailyTemperatures) : ℝ :=
  temps.maximum?.getD 0

/-- Calculates the range (difference between max and min) of temperatures -/
noncomputable def tempRange (temps : DailyTemperatures) : ℝ :=
  maxTemp temps - minTemp temps

theorem max_temp_range (temps : DailyTemperatures) :
  temps.length = numDays →
  average temps = 45 →
  minTemp temps = 28 →
  tempRange temps ≤ 119 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_temp_range_l793_79342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l793_79323

/-- Represents a cone with a lateral surface that unfolds into a semicircle -/
structure Cone where
  lateral_radius : ℝ
  lateral_radius_positive : lateral_radius > 0

/-- The height of a cone given its lateral surface radius -/
noncomputable def cone_height (c : Cone) : ℝ :=
  Real.sqrt (c.lateral_radius ^ 2 - (c.lateral_radius / Real.pi) ^ 2)

/-- The area of a cone's axial section given its lateral surface radius -/
noncomputable def cone_axial_section_area (c : Cone) : ℝ :=
  c.lateral_radius * cone_height c

theorem cone_properties (c : Cone) (h : c.lateral_radius = 2) :
  cone_height c = Real.sqrt 3 ∧ cone_axial_section_area c = Real.sqrt 3 := by
  sorry

#check cone_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l793_79323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l793_79335

theorem min_value_and_inequality (a b : ℝ) (x y z : ℝ) : 
  a > b → b > 0 → 
  (∀ a' b', a' > b' → b' > 0 → a' + 1 / ((a' - b') * b') ≥ 3) ∧ 
  (x + y + z = 3 → x^2 + 4*y^2 + z^2 = 3 → |x + 2*y + z| ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l793_79335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probability_l793_79391

-- Define a die as having 6 faces
def die : ℕ := 6

-- Define the probability of an event given favorable outcomes and total outcomes
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

-- Define the condition for the first roll
def first_roll_condition (n : ℕ) : Bool :=
  n < 3

-- Define the condition for the second roll
def second_roll_condition (n : ℕ) : Bool :=
  n > 3

-- State the theorem
theorem die_roll_probability :
  probability
    (Finset.filter (λ (pair : ℕ × ℕ) => first_roll_condition pair.fst ∧ second_roll_condition pair.snd)
      (Finset.product (Finset.range die) (Finset.range die))).card
    (die * die) = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probability_l793_79391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l793_79333

noncomputable def is_valid_point (x y : ℕ) : Prop :=
  0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10

noncomputable def angle_with_x_axis (x y : ℕ) : ℝ :=
  Real.arctan (y / x : ℝ)

noncomputable def triangle_area (x y : ℕ) : ℝ :=
  (x * y : ℝ) / 2

def four_digit_number (x₁ y₁ x₂ y₂ : ℕ) : ℕ :=
  x₁ * 1000 + x₂ * 100 + y₂ * 10 + y₁

theorem unique_four_digit_number (x₁ y₁ x₂ y₂ : ℕ) :
  is_valid_point x₁ y₁ ∧
  is_valid_point x₂ y₂ ∧
  angle_with_x_axis x₁ y₁ > π/4 ∧
  angle_with_x_axis x₂ y₂ < π/4 ∧
  triangle_area x₂ y₂ - triangle_area x₁ y₁ = 33.5 →
  four_digit_number x₁ y₁ x₂ y₂ = 1985 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l793_79333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_plan_solution_l793_79371

/-- Represents a cellular phone plan -/
structure PhonePlan where
  baseFee : ℚ
  baseMinutes : ℚ
  overageRate : ℚ

/-- Calculates the cost of a phone plan for a given number of minutes -/
def planCost (plan : PhonePlan) (minutes : ℚ) : ℚ :=
  plan.baseFee + max 0 (minutes - plan.baseMinutes) * plan.overageRate

/-- The solution to the phone plan problem -/
theorem phone_plan_solution (x : ℚ) :
  let plan1 := PhonePlan.mk 50 x (35/100)
  let plan2 := PhonePlan.mk 75 1000 (45/100)
  planCost plan1 2500 = planCost plan2 2500 → x = 500 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_plan_solution_l793_79371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l793_79309

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides 20 cm and 18 cm, 
    and height 13 cm, is 247 square centimeters -/
theorem trapezium_area_example : trapeziumArea 20 18 13 = 247 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l793_79309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l793_79346

-- Define the function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := -x^2 / Real.exp x + (b - 1) * x + a

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x + 2) / (2 * Real.exp x)

-- State the theorem
theorem tangent_and_inequality (a b : ℝ) :
  (∀ x, x ≠ 0 → (f x a b - f 0 a b) / x = 0 → False) →
  b = 1 ∧
  (a = 1 → ∀ x, x > 0 → f x a b > g x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l793_79346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_117_l793_79330

def is_divisor (d n : ℕ) : Prop := n % d = 0

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n) (Finset.range (n + 1))).sum id

def count_prime_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ Nat.Prime d) (Finset.range (n + 1))).card

theorem divisors_of_117 :
  sum_of_divisors 117 = 182 ∧ count_prime_divisors 117 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_117_l793_79330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_eventually_stable_l793_79359

/-- Represents a board state as a finite sequence of positive integers -/
def BoardState := List Nat

/-- The operation of replacing two elements with their lcm and gcd -/
def replace_pair (a b : Nat) : List Nat :=
  [Nat.lcm a b, Nat.gcd a b]

/-- Applies the replace_pair operation to a BoardState -/
def apply_operation (state : BoardState) : BoardState :=
  match state with
  | [] => []
  | [x] => [x]
  | x :: y :: rest => (replace_pair x y) ++ rest

/-- Predicate to check if a BoardState is stable (unchanged by further operations) -/
def is_stable (state : BoardState) : Prop :=
  apply_operation state = state

/-- Helper function to apply the operation multiple times -/
def apply_n_times (n : Nat) (state : BoardState) : BoardState :=
  match n with
  | 0 => state
  | n + 1 => apply_n_times n (apply_operation state)

theorem board_eventually_stable :
  ∀ (initial : BoardState), ∃ (n : Nat) (final : BoardState),
    (apply_n_times n initial = final) ∧ is_stable final :=
by
  sorry

#check board_eventually_stable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_eventually_stable_l793_79359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_l793_79340

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hr_to_m_per_s (v : ℝ) : ℝ := v * (1000 / 3600)

/-- Calculates the time (in seconds) for an object of given length to pass a stationary point -/
noncomputable def time_to_pass (length : ℝ) (speed_km_per_hr : ℝ) : ℝ :=
  length / km_per_hr_to_m_per_s speed_km_per_hr

theorem train_passing_tree (train_length : ℝ) (train_speed_km_per_hr : ℝ)
    (h1 : train_length = 240)
    (h2 : train_speed_km_per_hr = 108) :
    time_to_pass train_length train_speed_km_per_hr = 8 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_l793_79340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_in_range_l793_79300

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -Real.log x
  else if x > 1 then Real.log x
  else 0

def is_tangent_line (l : ℝ → ℝ) (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ k b, l = (λ x ↦ k * x + b) ∧ 
         l p.1 = f p.1 ∧
         (∀ h, h ≠ 0 → (l (p.1 + h) - l p.1) / h = (f (p.1 + h) - f p.1) / h)

def perpendicular (l₁ l₂ : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ k₁ k₂ b₁ b₂, 
    l₁ = (λ x ↦ k₁ * x + b₁) ∧
    l₂ = (λ x ↦ k₂ * x + b₂) ∧
    k₁ * k₂ = -1 ∧
    l₁ p.1 = l₂ p.1 ∧ l₁ p.1 = p.2

theorem area_of_triangle_PAB_in_range :
  ∀ (l₁ l₂ : ℝ → ℝ) (P₁ P₂ P A B : ℝ × ℝ),
    0 < P₁.1 ∧ P₁.1 < 1 →
    P₂.1 > 1 →
    is_tangent_line l₁ f P₁ →
    is_tangent_line l₂ f P₂ →
    perpendicular l₁ l₂ P →
    A.1 = 0 ∧ l₁ A.1 = A.2 →
    B.1 = 0 ∧ l₂ B.1 = B.2 →
    ∃ S, 0 < S ∧ S < 1 ∧ S = (1/2) * |A.2 - B.2| * |P.1| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_in_range_l793_79300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l793_79307

/-- The probability that two groups of tourists can contact each other -/
theorem tourist_contact_probability (p : ℝ) 
  (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  1 - (1 - p) ^ 40 = 1 - (1 - p) ^ 40 := by
  -- The proof is trivial as we're stating equality to itself
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l793_79307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l793_79356

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m / x

def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * m * x + m - 6

theorem range_of_m :
  {m : ℝ | (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∧
           ¬(∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0)} =
  {m : ℝ | -3 ≤ m ∧ m ≤ 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l793_79356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_different_sums_l793_79387

def Matrix3x3 := Fin 3 → Fin 3 → ℤ

def rowSum (m : Matrix3x3) (i : Fin 3) : ℤ := 
  (m i 0) + (m i 1) + (m i 2)

def colSum (m : Matrix3x3) (j : Fin 3) : ℤ := 
  (m 0 j) + (m 1 j) + (m 2 j)

def allSumsEqual (m : Matrix3x3) : Prop := 
  ∀ i j : Fin 3, rowSum m i = colSum m j

def numDifferentSums (m : Matrix3x3) : ℕ := 
  Finset.card (Finset.image (rowSum m) Finset.univ ∪ Finset.image (colSum m) Finset.univ)

def numChanges (m1 m2 : Matrix3x3) : ℕ := 
  Finset.card (Finset.filter (fun p => m1 p.1 p.2 ≠ m2 p.1 p.2) (Finset.product Finset.univ Finset.univ))

theorem min_changes_for_different_sums (m : Matrix3x3) (h : allSumsEqual m) :
  ∃ m' : Matrix3x3, numDifferentSums m' = 6 ∧ 
    (∀ m'' : Matrix3x3, numDifferentSums m'' = 6 → numChanges m m'' ≥ numChanges m m') ∧
    numChanges m m' = 4 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_different_sums_l793_79387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_2pi_3_l793_79385

theorem cos_2alpha_minus_2pi_3 (α : ℝ) (h : Real.sin (α + π/6) = 1/3) :
  Real.cos (2*α - 2*π/3) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_2pi_3_l793_79385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_exists_l793_79332

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define when a circle is tangent to a line
def is_tangent_to_line (c : Circle) (l : Line) : Prop :=
  ∃ p : Point, distance p c.center = c.radius ∧
    l.a * p.x + l.b * p.y + l.c = 0

-- Define when two circles are tangent
def are_circles_tangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

-- Define when a line and a circle do not intersect
def do_not_intersect (l : Line) (c : Circle) : Prop :=
  ∀ p : Point, l.a * p.x + l.b * p.y + l.c = 0 →
    distance p c.center > c.radius

-- The main theorem
theorem tangent_circle_exists (L : Line) (C : Circle) (r : ℝ) :
  r > 0 →
  do_not_intersect L C →
  ∃ C' : Circle, C'.radius = r ∧
    is_tangent_to_line C' L ∧
    are_circles_tangent C' C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_exists_l793_79332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_five_l793_79322

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci_like_sequence (n + 1) + fibonacci_like_sequence n

theorem fifth_term_is_five : fibonacci_like_sequence 4 = 5 := by
  rw [fibonacci_like_sequence]
  rw [fibonacci_like_sequence]
  rw [fibonacci_like_sequence]
  rw [fibonacci_like_sequence]
  rfl

#eval fibonacci_like_sequence 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_five_l793_79322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_is_negative_one_l793_79328

-- Define the parabola C: x^2 = 2y
def C (x y : ℝ) : Prop := x^2 = 2*y

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define the line passing through (-2,4)
def line (x y : ℝ) (k : ℝ) : Prop := y - 4 = k * (x + 2)

-- Define points A and B as intersections of the line and parabola C
def A (x y : ℝ) (k : ℝ) : Prop := C x y ∧ line x y k
def B (x y : ℝ) (k : ℝ) : Prop := C x y ∧ line x y k ∧ ¬(x = P.1 ∧ y = P.2)

-- Define slopes k1 and k2
noncomputable def k1 (x y : ℝ) : ℝ := (y - P.2) / (x - P.1)
noncomputable def k2 (x y : ℝ) : ℝ := (y - P.2) / (x - P.1)

theorem slopes_product_is_negative_one :
  ∀ (x1 y1 x2 y2 k : ℝ),
    A x1 y1 k → B x2 y2 k →
    k1 x1 y1 * k2 x2 y2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_is_negative_one_l793_79328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l793_79360

/-- The polar equation of curve C₁ -/
def C₁ (ρ θ : ℝ) : Prop :=
  ρ > 0 ∧ 3 * ρ^2 = 12 * ρ * Real.cos θ - 10

/-- The Cartesian equation of curve C₂ -/
def C₂ (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

/-- The distance between two points in 2D space -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_between_curves :
  ∃ (min_dist : ℝ),
    min_dist = Real.sqrt (2/3) ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      (∃ ρ θ, C₁ ρ θ ∧ x₁ = ρ * Real.cos θ ∧ y₁ = ρ * Real.sin θ) →
      C₂ x₂ y₂ →
      distance x₁ y₁ x₂ y₂ ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l793_79360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l793_79362

/-- Represents the speed of a car in miles per hour -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Theorem stating that a car traveling 520 miles in 8 hours has a speed of 65 miles per hour -/
theorem car_speed : speed 520 8 = 65 := by
  -- Unfold the definition of speed
  unfold speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l793_79362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_is_ten_l793_79318

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem sum_of_first_five_terms_is_ten
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 2) :
  sum_of_arithmetic_sequence a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_is_ten_l793_79318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_l793_79350

theorem cube_root_of_product : (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_l793_79350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l793_79331

/-- Given a circular piece of paper with radius r, if a cone is formed from a sector of this paper
    with radius 10 cm and volume 300π cubic cm, then the central angle of the unused sector is
    approximately 92.05° -/
theorem unused_sector_angle (r : ℝ) : 
  r > 0 → ∃ (h : ℝ), 
    h > 0 ∧ 
    (1 / 3 : ℝ) * π * 10^2 * h = 300 * π ∧ 
    r^2 = 10^2 + h^2 ∧ 
    ∃ (θ : ℝ), 
      θ > 0 ∧ 
      θ < 360 ∧ 
      (20 * π) / (2 * π * r) * 360 = θ ∧ 
      |360 - θ - 92.05| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l793_79331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l793_79319

-- Define the necessary structures
structure Line : Type
structure Plane : Type

-- Define the relationships
def parallel (l : Line) (p : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_intersection 
  (l : Line) (α : Plane) 
  (h1 : ¬ parallel l α) 
  (h2 : ¬ contained_in l α) : 
  ¬ ∃ (m : Line), line_in_plane m α ∧ parallel_lines l m :=
by
  sorry

#check line_plane_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l793_79319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l793_79303

/-- The general term of the sequence -/
def a (n : ℕ) (lambda : ℝ) : ℝ := n^2 - (6 + 2*lambda)*n + 2016

/-- Predicate for a_6 or a_7 being the minimum term of the sequence -/
def is_min_at_6_or_7 (lambda : ℝ) : Prop :=
  ∀ n : ℕ, (a 6 lambda ≤ a n lambda) ∨ (a 7 lambda ≤ a n lambda)

/-- The main theorem -/
theorem lambda_range (lambda : ℝ) :
  is_min_at_6_or_7 lambda ↔ (5/2 < lambda ∧ lambda < 9/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l793_79303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l793_79365

-- Define propositions p and q
def p : Prop := ∃ x : ℝ, Real.sin x = Real.pi / 2
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- State the theorem
theorem problem_statement : (¬p ∧ q) ∧ ¬(p ∧ ¬q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l793_79365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l793_79361

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 4/5) : 
  Real.tan (2 * x) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l793_79361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_range_l793_79389

theorem increasing_log_function_range (u : ℝ) (a : ℝ) :
  (0 < u ∧ u ≠ 1) →
  (∀ x ∈ Set.Icc (3/2 : ℝ) 2, StrictMono (fun x => Real.log (6*a*x^2 - 2*x + 3))) →
  a ∈ Set.Ioo (1/24 : ℝ) (1/12) ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_range_l793_79389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_of_a_l793_79393

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x
def g (a : ℝ) (x : ℝ) := a * x
noncomputable def F (a : ℝ) (x : ℝ) := f x - g a x

theorem min_value_and_range_of_a :
  (∃ (m : ℝ), m = 3 ∧
    ∀ (x₁ x₂ : ℝ), x₁ ≥ 0 → x₂ ≥ 0 → f x₁ = g (1/3) x₂ → x₂ - x₁ ≥ m) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ≥ 0 → F a x ≥ F a (-x)) ↔ a ≤ 2) :=
by sorry

#check min_value_and_range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_of_a_l793_79393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_eddy_freddy_l793_79313

/-- Represents a traveler with their journey details -/
structure Traveler where
  name : String
  distance : ℚ
  time : ℚ

/-- Calculates the average speed of a traveler -/
def averageSpeed (t : Traveler) : ℚ :=
  t.distance / t.time

/-- Theorem stating the ratio of average speeds between Eddy and Freddy -/
theorem speed_ratio_eddy_freddy (eddy freddy : Traveler)
    (h1 : eddy.name = "Eddy")
    (h2 : freddy.name = "Freddy")
    (h3 : eddy.distance = 600)
    (h4 : freddy.distance = 300)
    (h5 : eddy.time = 3)
    (h6 : freddy.time = 3) :
    averageSpeed eddy / averageSpeed freddy = 2 := by
  sorry

#check speed_ratio_eddy_freddy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_eddy_freddy_l793_79313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_representation_l793_79305

theorem irreducible_fraction_representation (p q : ℕ) 
  (hp : p > 0)
  (hq : q > 0)
  (h_odd : Odd q)
  (h_irreducible : ∀ (d : ℕ), d ∣ p ∧ d ∣ q → d = 1 ∨ d = Nat.gcd p q) :
  ∃ (n k : ℕ), n > 0 ∧ k > 0 ∧ (p : ℚ) / q = (n : ℚ) / (2^k - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_representation_l793_79305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l793_79324

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r : ℝ) (θ : ℝ) (z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ :=
  (10, Real.pi / 3 + Real.pi / 6, 2)

/-- The expected point in rectangular coordinates -/
def rectangular_point : ℝ × ℝ × ℝ :=
  (0, 10, 2)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l793_79324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l793_79376

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + 2 * x * k

-- State the theorem
theorem f_derivative_at_one :
  ∃ (k : ℝ), (∀ x, f k x = (1/3) * x^3 + 2 * x * k) ∧
             (deriv (f k) 1 = k) ∧
             (k = -1) := by
  -- We'll use k = -1 as our witness
  use -1
  constructor
  · -- First part: ∀ x, f k x = (1/3) * x^3 + 2 * x * k
    intro x
    rfl
  constructor
  · -- Second part: deriv (f k) 1 = k
    sorry -- This requires calculus, which we'll skip for now
  · -- Third part: k = -1
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l793_79376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_main_theorem_l793_79373

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the line passing through the center (3, 5)
def line_eq (k : ℝ) (x y : ℝ) : Prop := y - 5 = k * (x - 3)

-- Define the intersection points
noncomputable def intersectCircleLine (k : ℝ) : ℝ × ℝ × ℝ × ℝ := by
  let x₁ := 3 - Real.sqrt (5 / (1 + k^2))
  let y₁ := 5 + k * (x₁ - 3)
  let x₂ := 3 + Real.sqrt (5 / (1 + k^2))
  let y₂ := 5 + k * (x₂ - 3)
  exact (x₁, y₁, x₂, y₂)

-- Define the y-axis intersection point
def intersectYAxis (k : ℝ) : ℝ × ℝ := (0, 5 - 3*k)

-- A is midpoint of PB condition
def isMidpoint (k : ℝ) : Prop := by
  let (x₁, y₁, x₂, y₂) := intersectCircleLine k
  let (px, py) := intersectYAxis k
  exact 2 * x₁ = px + x₂ ∧ 2 * y₁ = py + y₂

theorem line_equation (k : ℝ) :
  isMidpoint k → (k = 2 ∨ k = -2) := by
  sorry

theorem main_theorem :
  ∃ k : ℝ, isMidpoint k ∧ (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_main_theorem_l793_79373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l793_79388

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_properties
  (seq : ArithmeticSequence)
  (h1 : seq.a 1 > 0)
  (h2 : -1 < seq.a 7 / seq.a 6)
  (h3 : seq.a 7 / seq.a 6 < 0) :
  seq.d < 0 ∧ (∀ n > 12, S seq n ≤ 0) ∧ S seq 12 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l793_79388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_area_difference_l793_79382

-- Define the properties of the largest room (trapezoid)
noncomputable def largest_room_side1 : ℝ := 45
noncomputable def largest_room_side2 : ℝ := 30
noncomputable def largest_room_height : ℝ := 25

-- Define the properties of the smallest room (parallelogram)
noncomputable def smallest_room_base : ℝ := 15
noncomputable def smallest_room_height : ℝ := 8

-- Define the area of a trapezoid
noncomputable def trapezoid_area (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Define the area of a parallelogram
noncomputable def parallelogram_area (b h : ℝ) : ℝ := b * h

-- Theorem statement
theorem room_area_difference :
  trapezoid_area largest_room_side1 largest_room_side2 largest_room_height -
  parallelogram_area smallest_room_base smallest_room_height = 817.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_area_difference_l793_79382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_opposite_vertex_probability_l793_79374

structure Octahedron :=
  (vertices : Finset (Fin 6))
  (edges : Finset (Fin 6 × Fin 6))
  (opposite : Fin 6 → Fin 6)

def ant_walk (o : Octahedron) (start : Fin 6) (steps : ℕ) : Finset (Fin 6) :=
  sorry

theorem ant_opposite_vertex_probability 
  (o : Octahedron) 
  (start : Fin 6) : 
  (1 : ℚ) / 128 = Finset.card (ant_walk o start 6 ∩ {o.opposite start}) / Finset.card (ant_walk o start 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_opposite_vertex_probability_l793_79374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l793_79353

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let asymptote₁ : ℝ → ℝ → Prop := λ x y => b * x + a * y = 0
  let asymptote₂ : ℝ → ℝ → Prop := λ x y => b * x - a * y = 0
  let circle : ℝ → ℝ → Prop := λ x y => x^2 + (y - 4)^2 = 4
  (∀ x y, asymptote₁ x y → circle x y) →
  (∀ x y, asymptote₂ x y → circle x y) →
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l793_79353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_pies_are_fewest_l793_79351

-- Define the shapes of pies
inductive PieShape
  | Rectangle
  | Square
  | Circle
  | Triangle

-- Define a function to calculate the area of a pie given its shape
noncomputable def pieArea (shape : PieShape) : ℝ :=
  match shape with
  | PieShape.Rectangle => 24  -- 4 * 6
  | PieShape.Square => 16     -- 4 * 4
  | PieShape.Circle => Real.pi * 9  -- π * r^2, r = 3
  | PieShape.Triangle => 24   -- (1/2) * 6 * 8

-- Define a function to calculate the number of pies given the total dough amount and pie shape
noncomputable def numberOfPies (totalDough : ℝ) (shape : PieShape) : ℝ :=
  totalDough / pieArea shape

-- Theorem statement
theorem circular_pies_are_fewest (totalDough : ℝ) (h : totalDough > 0) :
  ∀ shape, shape ≠ PieShape.Circle →
    numberOfPies totalDough PieShape.Circle < numberOfPies totalDough shape := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_pies_are_fewest_l793_79351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_nature_l793_79349

theorem roots_nature : 
  ∃! r : ℝ, r^2 - 4*r*Real.sqrt 2 + 8 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_nature_l793_79349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l793_79326

/-- A monic cubic polynomial -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

/-- Checks if a polynomial is monic cubic -/
def Polynomial.IsMonicCubic (p : ℝ → ℝ) : Prop :=
  ∃ a b c, p = fun x ↦ x^3 + a*x^2 + b*x + c

/-- Sum of roots of a polynomial -/
noncomputable def SumOfRoots (p : ℝ → ℝ) : ℝ := sorry

/-- Sum of cubes of roots of a polynomial -/
noncomputable def SumOfCubesOfRoots (p : ℝ → ℝ) : ℝ := sorry

theorem sum_of_cubes_of_roots (a b c : ℝ) :
  Polynomial.IsMonicCubic (MonicCubicPolynomial a b c) ∧
  (1 + a + b + c = 5) ∧
  (SumOfRoots (MonicCubicPolynomial a b c) = 1) →
  |SumOfCubesOfRoots (MonicCubicPolynomial a b c)| = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l793_79326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_meeting_l793_79321

/-- Represents the duration of the mathematicians' stay in minutes -/
noncomputable def m (a b c : ℕ) : ℝ := a - b * Real.sqrt (c : ℝ)

/-- Represents the probability of the mathematicians meeting -/
noncomputable def meeting_probability (m : ℝ) : ℝ := 
  (120 * 120 - (120 - m) * (120 - m)) / (120 * 120)

theorem mathematicians_meeting 
  (a b c : ℕ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hc_square_free : ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c)) 
  (hm : m a b c > 0) 
  (hprob : meeting_probability (m a b c) = 1/2) : 
  a + b + c = 182 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_meeting_l793_79321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brooke_homework_time_l793_79357

/-- Calculates the total time to complete homework given the number of problems and time per problem for each subject. -/
noncomputable def total_homework_time (math_problems : ℕ) (math_time : ℚ) (social_problems : ℕ) (social_time : ℚ) (science_problems : ℕ) (science_time : ℚ) : ℚ :=
  (math_problems : ℚ) * math_time + (social_problems : ℚ) * social_time / 60 + (science_problems : ℚ) * science_time

/-- Theorem stating that the total homework time for Brooke is 48 minutes. -/
theorem brooke_homework_time :
  total_homework_time 15 2 6 (1/2) 10 (3/2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brooke_homework_time_l793_79357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l793_79317

noncomputable section

-- Define the function f
def f (b c x : ℝ) : ℝ := 2 * x + b / x + c

-- State the theorem
theorem function_properties :
  ∃ (b c : ℝ),
    (f b c 1 = 4) ∧
    (f b c 2 = 5) ∧
    (b = 2 ∧ c = 0) ∧
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f b c x₁ > f b c x₂) ∧
    (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f b c x₁ < f b c x₂) ∧
    (∀ x m : ℝ, 1/2 ≤ x ∧ x ≤ 3 →
      (1/2 * f b c x + 4 * m < 1/2 * f b c (-x) + m^2 + 4 →
        m < 0 ∨ m > 4)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l793_79317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l793_79379

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The slope angle of the tangent line at a given point -/
noncomputable def slope_angle (x : ℝ) : ℝ := Real.arctan (f' x)

theorem tangent_slope_angle_at_one :
  slope_angle 1 = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l793_79379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l793_79354

theorem complex_fraction_simplification : 
  (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l793_79354
