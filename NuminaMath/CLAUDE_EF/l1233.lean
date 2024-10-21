import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nthMonomial_eq_monomialSeries_l1233_123377

/-- The nth monomial in the series -2x, 4x^3, -8x^5, 16x^7, ... -/
def nthMonomial (n : ℕ+) (x : ℝ) : ℝ := (-1)^(n:ℕ) * 2^(n:ℕ) * x^(2*(n:ℕ) - 1)

/-- The series of monomials -2x, 4x^3, -8x^5, 16x^7, ... -/
def monomialSeries (n : ℕ+) (x : ℝ) : ℝ := 
  match n with
  | 1 => -2 * x
  | 2 => 4 * x^3
  | 3 => -8 * x^5
  | 4 => 16 * x^7
  | _ => nthMonomial n x

theorem nthMonomial_eq_monomialSeries (n : ℕ+) (x : ℝ) : 
  nthMonomial n x = monomialSeries n x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nthMonomial_eq_monomialSeries_l1233_123377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l1233_123336

theorem solution_pairs : 
  {(x, y) : ℝ × ℝ | x^2 + y^2 + x + y = x*y*(x + y) - 10/27 ∧ |x*y| ≤ 25/9} = 
  {(5/3, 5/3), (-1/3, -1/3)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l1233_123336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1233_123306

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  if x ∈ Set.Ioo 0 3 then Real.log (2*x^2 - x + m) / Real.log 2 else 0

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def has_five_zeros_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), S.card = 5 ∧ (∀ x ∈ S, x ∈ Set.Icc (-3) 3 ∧ f x = 0) ∧
  (∀ x, x ∈ Set.Icc (-3) 3 ∧ f x = 0 → x ∈ S)

theorem f_range_theorem (m : ℝ) : 
  (is_odd_function (f m) ∧ 
   has_period (f m) 6 ∧ 
   has_five_zeros_in_interval (f m)) ↔ 
  m ∈ Set.Ioo (1/8) 1 ∪ {9/8} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1233_123306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_three_points_l1233_123308

/-- A line passes through three points: (2,3), (-4, m), and (-12, -1).
    This theorem proves that m = 9/7 for these points to be collinear. -/
theorem line_through_three_points (m : ℚ) : 
  (m - 3) / (-6 : ℚ) = ((-1) - m) / (-8 : ℚ) → m = 9/7 :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_three_points_l1233_123308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gain_percentage_l1233_123354

theorem total_gain_percentage (a_buy a_sell b_buy b_sell c_buy c_sell : ℚ) : 
  a_buy = 20 → a_sell = 25 →
  b_buy = 30 → b_sell = 35 →
  c_buy = 40 → c_sell = 60 →
  (((a_sell + b_sell + c_sell) - (a_buy + b_buy + c_buy)) / (a_buy + b_buy + c_buy)) * 100 = 33.33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gain_percentage_l1233_123354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_z_is_zero_l1233_123321

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 7*a + 6)

-- (1) z is real iff a = 1 or a = 6
theorem z_is_real (a : ℝ) : z a ∈ Set.range Complex.ofReal ↔ a = 1 ∨ a = 6 := by sorry

-- (2) z is purely imaginary iff a = -2
theorem z_is_purely_imaginary (a : ℝ) : (z a).re = 0 ∧ (z a).im ≠ 0 ↔ a = -2 := by sorry

-- (3) z is zero iff a = 1
theorem z_is_zero (a : ℝ) : z a = 0 ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_z_is_zero_l1233_123321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1233_123322

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2^x - 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1233_123322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1233_123303

/-- Given two points and a segment longer than their distance, 
    prove that the locus of points with constant sum of distances
    lies on one side of a specific line -/
theorem ellipse_property (F₁ F₂ M : ℝ × ℝ) (s : ℝ) : 
  s > Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) + 
  Real.sqrt ((M.2 - F₂.1)^2 + (M.2 - F₂.2)^2) = s →
  ∃ (e : Set (ℝ × ℝ)), 
    (∀ (P : ℝ × ℝ), P ∈ e ↔ 
      (P.1 - M.1) * ((F₁.1 - M.1) + (F₂.1 - M.1)) + 
      (P.2 - M.2) * ((F₁.2 - M.2) + (F₂.2 - M.2)) = 0) ∧
    (∀ (M' : ℝ × ℝ), 
      Real.sqrt ((M'.1 - F₁.1)^2 + (M'.2 - F₁.2)^2) + 
      Real.sqrt ((M'.2 - F₂.1)^2 + (M'.2 - F₂.2)^2) = s → 
      ((M'.1 - M.1) * ((F₁.1 - M.1) + (F₂.1 - M.1)) + 
       (M'.2 - M.2) * ((F₁.2 - M.2) + (F₂.2 - M.2)) > 0) = 
      ((F₁.1 - M.1) * ((F₁.1 - M.1) + (F₂.1 - M.1)) + 
       (F₁.2 - M.2) * ((F₁.2 - M.2) + (F₂.2 - M.2)) > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1233_123303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_square_walk_l1233_123372

/-- The area of the region visible to a person walking around a square with given side length and visibility radius -/
noncomputable def visible_area (side_length : ℝ) (visibility_radius : ℝ) : ℝ :=
  let inner_area := side_length^2 - (side_length - 2 * visibility_radius)^2
  let outer_rectangles := 4 * side_length * visibility_radius
  let outer_circles := Real.pi * visibility_radius^2
  inner_area + outer_rectangles + outer_circles

/-- The rounded area of the visible region -/
noncomputable def rounded_visible_area (side_length : ℝ) (visibility_radius : ℝ) : ℤ :=
  round (visible_area side_length visibility_radius)

theorem visible_area_square_walk :
  rounded_visible_area 7 2 = 109 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_square_walk_l1233_123372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_investment_duration_l1233_123391

/-- Represents a partner in the investment scenario -/
structure Partner where
  investment : ℚ
  duration : ℚ
  profit : ℚ

/-- The investment scenario with two partners -/
structure InvestmentScenario where
  p : Partner
  q : Partner
  investment_ratio : ℚ × ℚ
  profit_ratio : ℚ × ℚ

/-- Calculates the investment duration for partner Q given the scenario -/
noncomputable def calculate_q_duration (scenario : InvestmentScenario) : ℚ :=
  (scenario.p.investment * scenario.p.duration * scenario.profit_ratio.snd) /
  (scenario.q.investment * scenario.profit_ratio.fst)

/-- Theorem stating that Q's investment duration is 40 months -/
theorem q_investment_duration (scenario : InvestmentScenario) 
  (h1 : scenario.investment_ratio = (7, 5))
  (h2 : scenario.profit_ratio = (7, 10))
  (h3 : scenario.p.duration = 20)
  (h4 : scenario.p.investment / scenario.q.investment = 7 / 5) :
  calculate_q_duration scenario = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_investment_duration_l1233_123391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_min_fourth_quarter_score_l1233_123323

/-- Represents a student's scores for four quarters -/
structure QuarterlyScores where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- Calculates the average of four quarterly scores -/
noncomputable def average (scores : QuarterlyScores) : ℝ :=
  (scores.q1 + scores.q2 + scores.q3 + scores.q4) / 4

/-- The minimum average required for eligibility -/
def min_eligible_average : ℝ := 85

/-- Jamie's scores for the first three quarters -/
def jamie_first_three : ℝ × ℝ × ℝ := (84, 80, 83)

/-- Theorem: The minimum score Jamie needs in the 4th quarter for eligibility is 93% -/
theorem jamie_min_fourth_quarter_score :
  let (q1, q2, q3) := jamie_first_three
  ∀ q4 : ℝ,
    (average ⟨q1, q2, q3, q4⟩ ≥ min_eligible_average) ↔ (q4 ≥ 93) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_min_fourth_quarter_score_l1233_123323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l1233_123325

theorem simplify_sqrt_sum : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l1233_123325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1233_123340

-- Define the distances and times
noncomputable def distance_AB : ℝ := 600
noncomputable def distance_AC : ℝ := 360
noncomputable def time_Eddy : ℝ := 3
noncomputable def time_Freddy : ℝ := 4

-- Define the average speeds
noncomputable def speed_Eddy : ℝ := distance_AB / time_Eddy
noncomputable def speed_Freddy : ℝ := distance_AC / time_Freddy

-- Theorem to prove
theorem speed_ratio : 
  (speed_Eddy / speed_Freddy) = (20 : ℝ) / (9 : ℝ) := by
  -- Expand the definitions
  unfold speed_Eddy speed_Freddy
  -- Simplify the fraction
  simp [distance_AB, distance_AC, time_Eddy, time_Freddy]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1233_123340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnet_sticker_price_ratio_l1233_123384

/-- The price of the magnet in dollars -/
noncomputable def magnet_price : ℝ := 3

/-- The price of a single stuffed animal in dollars -/
noncomputable def stuffed_animal_price : ℝ := 6

/-- The price of the sticker in dollars -/
noncomputable def sticker_price : ℝ := magnet_price / 4

/-- The ratio of the magnet price to the sticker price -/
noncomputable def magnet_sticker_ratio : ℝ := magnet_price / sticker_price

theorem magnet_sticker_price_ratio :
  magnet_price = (1 / 4) * (2 * stuffed_animal_price) →
  magnet_sticker_ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnet_sticker_price_ratio_l1233_123384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1233_123369

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the point P
def point_P : ℝ × ℝ := (5, 5)

-- Define a line passing through point P
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y - point_P.2 = k * (x - point_P.1)

-- Define the chord length
noncomputable def chord_length (k : ℝ) : ℝ := 4

-- Theorem statement
theorem line_equation :
  ∃ k : ℝ, (∀ x y : ℝ, line_through_P k x y ∧ my_circle x y → chord_length k = 4) →
  (∀ x y : ℝ, line_through_P k x y ↔ (2 * x - y - 5 = 0 ∨ x - 2 * y + 5 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1233_123369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_minutes_correct_l1233_123356

/-- Represents the initial distribution of ants on the number line -/
def initial_distribution : List (Nat × Nat) :=
  [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

/-- The total number of ants -/
def total_ants : Nat := 25

/-- The minimum number of minutes required to distribute ants -/
def min_minutes : Nat := 250

/-- Proves that the minimum number of minutes to distribute ants is correct -/
theorem min_minutes_correct (move_strategy : List (Nat × Nat)) :
  (∀ (pos : Nat), pos ∈ move_strategy.map Prod.fst → pos ≤ 25) →
  move_strategy.length = total_ants →
  (∀ (pos : Nat), pos ∈ move_strategy.map Prod.fst → 
    (move_strategy.filter (λ p => p.fst = pos)).length = 1) →
  (∀ (init_pos : Nat) (final_pos : Nat),
    init_pos ∈ initial_distribution.map Prod.fst ∧
    final_pos ∈ move_strategy.map Prod.fst →
    ∃ (ant : Nat), ant ≤ 5 ∧
    (Int.natAbs (init_pos - final_pos) : Nat) ≤ min_minutes) →
  min_minutes = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_minutes_correct_l1233_123356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seismic_prediction_accuracy_l1233_123362

theorem seismic_prediction_accuracy :
  ∀ (total_days : ℝ) (quiet_days : ℝ) (predicted_quiet : ℝ) (correct_quiet_predictions : ℝ),
    quiet_days = 0.8 * total_days →
    predicted_quiet = 0.64 * total_days →
    correct_quiet_predictions = 0.7 * quiet_days →
    let active_days := total_days - quiet_days
    let incorrect_active_predictions := predicted_quiet - correct_quiet_predictions
    incorrect_active_predictions / active_days = 0.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seismic_prediction_accuracy_l1233_123362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_sequences_ten_flips_five_heads_l1233_123326

theorem coin_flip_sequences (n : ℕ) (k : ℕ) (h : k ≤ n) :
  (Nat.choose n k) = (Finset.filter (fun s => s.card = k) (Finset.powerset (Finset.range n))).card :=
by sorry

theorem ten_flips_five_heads :
  (Nat.choose 10 5) = 252 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_sequences_ten_flips_five_heads_l1233_123326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_f_range_l1233_123368

noncomputable def f (x : ℝ) : ℝ := (2^(x+1) - 1) / (2^x + 1)

theorem floor_f_range : ∀ x : ℝ, ⌊f x⌋ ∈ ({-1, 0, 1} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_f_range_l1233_123368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_or_fourth_quadrant_l1233_123320

theorem angle_in_second_or_fourth_quadrant (θ : Real) :
  (Real.cos θ + Real.sin θ = 1/2) →
  (π/2 < θ % (2*π) ∧ θ % (2*π) < π) ∨ (3*π/2 < θ % (2*π) ∧ θ % (2*π) < 2*π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_or_fourth_quadrant_l1233_123320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l1233_123345

/-- The selling price of an article given its cost price and potential discount and profit percentages. -/
noncomputable def selling_price (cost_price : ℝ) (discount_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  cost_price * (1 + profit_percent / 100) / (1 - discount_percent / 100)

/-- Theorem stating that the selling price is 12000 given the conditions of the problem. -/
theorem article_selling_price : 
  let cost_price : ℝ := 10000
  let discount_percent : ℝ := 10
  let profit_percent : ℝ := 8
  selling_price cost_price discount_percent profit_percent = 12000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l1233_123345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l1233_123312

/-- Given four distinct points P, A, B, C in a vector space V, 
    if PA + PB + PC = 0 and AB + AC = m * AP, then m = 3 -/
theorem centroid_property {V : Type*} [AddCommGroup V] [Module ℝ V]
  (P A B C : V) (m : ℝ) 
  (h1 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (h2 : (A - P) + (B - P) + (C - P) = 0) 
  (h3 : (B - A) + (C - A) = m • (A - P)) : 
  m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l1233_123312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l1233_123317

/-- Represents the number of eighth graders -/
def e : ℕ := sorry

/-- Average minutes run by sixth graders per day -/
def sixth_grade_avg : ℚ := 20

/-- Average minutes run by seventh graders per day -/
def seventh_grade_avg : ℚ := 18

/-- Average minutes run by eighth graders per day -/
def eighth_grade_avg : ℚ := 22

/-- Number of sixth graders -/
def sixth_grade_count : ℕ := 3 * e

/-- Number of seventh graders -/
def seventh_grade_count : ℕ := 2 * e

/-- Number of eighth graders -/
def eighth_grade_count : ℕ := e

/-- Total minutes run by all students -/
def total_minutes : ℚ :=
  sixth_grade_avg * (sixth_grade_count : ℚ) +
  seventh_grade_avg * (seventh_grade_count : ℚ) +
  eighth_grade_avg * (eighth_grade_count : ℚ)

/-- Total number of students -/
def total_students : ℕ :=
  sixth_grade_count + seventh_grade_count + eighth_grade_count

/-- Theorem: The average number of minutes run per day by all students is 118/6 -/
theorem average_minutes_run :
  total_minutes / (total_students : ℚ) = 118 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l1233_123317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1233_123310

/-- An ellipse with the given properties has a major axis of length 8 -/
theorem ellipse_major_axis_length :
  ∀ (E : Set (ℝ × ℝ)),
  (∃ (x y : ℝ), (x, 0) ∈ E ∧ (0, y) ∈ E) →  -- tangent to x-axis and y-axis
  ((3, -4 + Real.sqrt 8) ∈ E ∧ (3, -4 - Real.sqrt 8) ∈ E) →  -- foci locations
  (∃ (a b : ℝ × ℝ), a ∈ E ∧ b ∈ E ∧ dist a b = 8) :=  -- major axis length
by sorry

/-- Helper function to calculate distance between two points -/
noncomputable def dist (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1233_123310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shift_l1233_123346

noncomputable def f (x : ℝ) := Real.sin (4 * x - Real.pi / 2) + 2

theorem phase_and_vertical_shift :
  (∃ (p : ℝ), ∀ (x : ℝ), f x = Real.sin (4 * (x - p))) ∧
  (∃ (v : ℝ), ∀ (x : ℝ), f x = Real.sin (4 * x - Real.pi / 2) + v) ∧
  (∀ (x : ℝ), f x = Real.sin (4 * (x - Real.pi / 8)) + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shift_l1233_123346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_width_ratio_l1233_123383

/-- Represents a rectangular sheet of paper with a frame -/
structure FramedSheet where
  /-- Length of the shorter side of the sheet -/
  short_side : ℝ
  /-- Width of the frame -/
  frame_width : ℝ

/-- The ratio of the longer side to the shorter side of the sheet -/
noncomputable def side_ratio : ℝ := Real.sqrt 2

/-- The area of the entire sheet -/
noncomputable def sheet_area (s : FramedSheet) : ℝ := s.short_side * s.short_side * side_ratio

/-- The area of the frame -/
noncomputable def frame_area (s : FramedSheet) : ℝ :=
  sheet_area s - (s.short_side - 2 * s.frame_width) * (s.short_side * side_ratio - 2 * s.frame_width)

/-- The theorem stating the relationship between the frame width and the shorter side -/
theorem frame_width_ratio (s : FramedSheet) :
  frame_area s = (1/2) * sheet_area s →
  s.frame_width / s.short_side = (1 + Real.sqrt 2 - Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_width_ratio_l1233_123383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_shore_l1233_123301

/-- Represents the probability of a bridge being destroyed -/
noncomputable def p : ℝ := 1/2

/-- Represents the probability of a bridge remaining intact -/
noncomputable def q : ℝ := 1 - p

/-- Represents the infinite series of probabilities for all possible paths -/
noncomputable def probabilitySeries : ℕ → ℝ := fun n => p^n * q^(n+1)

/-- Theorem stating the probability of reaching the shore from the first island -/
theorem probability_reach_shore :
  (∑' n, probabilitySeries n) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_shore_l1233_123301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_count_l1233_123300

def digits : List ℕ := [2, 0, 1, 3]

def is_valid_permutation (perm : List ℕ) : Bool :=
  perm.length = 4 && perm.head? ≠ some 0 && perm.toFinset = digits.toFinset

def count_valid_permutations : ℕ := (digits.permutations.filter is_valid_permutation).length

theorem rearrangement_count : count_valid_permutations = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_count_l1233_123300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1233_123335

theorem trigonometric_equation_solution (t : ℝ) :
  (Real.cos t ≠ 0) →
  (Real.sin t ≠ 0) →
  (2 * Real.sin (2 * t) - Real.sin (4 * t) ≠ 0) →
  (Real.tan t ^ 2 - (2 * Real.sin (2 * t) + Real.sin (4 * t)) / (2 * Real.sin (2 * t) - Real.sin (4 * t)) = 2 * (1 / Real.tan (2 * t))) →
  ∃ k : ℤ, t = π / 4 * (2 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1233_123335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_guarantee_square_l1233_123397

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def add_digits (initial : ℕ) (digit_sequence : Fin n → Fin 10) : ℕ :=
  sorry

def can_form_square (initial : ℕ) : Prop :=
  ∃ (final : ℕ), is_perfect_square final ∧
  ∃ (steps : ℕ), ∃ (digit_sequence : Fin steps → Fin 10),
    final = add_digits initial digit_sequence

theorem cannot_guarantee_square :
  ¬ ∀ (moves : ℕ), can_form_square 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_guarantee_square_l1233_123397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1233_123302

noncomputable def f (x : ℝ) := Real.sqrt (4 - Real.sqrt (6 - Real.sqrt x))

theorem f_domain : Set.Icc (0 : ℝ) 36 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1233_123302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antons_card_is_seven_l1233_123319

def original_cards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def remaining_cards : Finset ℕ := {1, 4, 5, 8}

def valid_selection (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ s ⊆ original_cards ∧
  ∃ (a b c d : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b = c * d

theorem antons_card_is_seven :
  ∃ (s : Finset ℕ), valid_selection s ∧
  (s.erase ((original_cards \ remaining_cards).min.getD 0) = remaining_cards) →
  (original_cards \ remaining_cards).min = some 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antons_card_is_seven_l1233_123319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1233_123359

/-- The function f as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x) * Real.cos (ω * x) - Real.sqrt 3 / 2 + Real.sqrt 3 * (Real.cos (ω * x))^2

/-- The theorem statement -/
theorem triangle_area (ω : ℝ) (A B C : ℝ) (a b c : ℝ) : 
  ω > 0 → 
  (∀ x, f ω (x + π) = f ω x) →  -- Smallest positive period is π
  0 < A → A < π/2 →  -- A is acute
  f ω A = 0 →
  a = 1 →
  b + c = 2 →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →  -- Cosine rule
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1233_123359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_two_l1233_123328

/-- A function f(x) with specific properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

/-- The theorem stating the derivative of f at x = 2 -/
theorem f_derivative_at_two (a b : ℝ) :
  (f a b 1 = -2) →
  (deriv (f a b) 1 = 0) →
  (deriv (f a b) 2 = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_two_l1233_123328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_proof_l1233_123385

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement -/
theorem principal_amount_proof :
  let final_amount : ℝ := 4410
  let interest_rate : ℝ := 0.05
  let time_period : ℝ := 2
  let principal : ℝ := 4000
  compound_interest principal interest_rate time_period = final_amount :=
by
  -- Unfold the definition of compound_interest
  unfold compound_interest
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_proof_l1233_123385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_l1233_123388

noncomputable def x (n : ℕ) (y : ℝ) : ℝ := 
  match n with
  | 0 => (y - 1) / (y + 1)
  | n + 1 => (x n y - 1) / (x n y + 1)

theorem sequence_converges (y : ℝ) (h : y ≠ -1) : 
  x 1977 y = -1/3 → y = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_l1233_123388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_odd_l1233_123338

def sequenceProperty (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 2 = 7 ∧
  ∀ n > 1, ∃ k : ℤ, a (n + 1) = k ∧
    (-1/2 : ℚ) < (k : ℚ) - (-(a n)^2 / (a (n-1))^2 : ℚ) - 1/2

theorem sequence_odd (a : ℕ → ℤ) (h : sequenceProperty a) :
  ∀ n > 1, Odd (a n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_odd_l1233_123338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1233_123379

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line x-ay-2=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := 1/a

/-- The slope of the line 2ax-(a-3)y+1=0 -/
noncomputable def slope2 (a : ℝ) : ℝ := (a-3)/(2*a)

/-- The condition a=1 is sufficient but not necessary for perpendicularity -/
theorem perpendicular_condition (a : ℝ) :
  (a = 1 → are_perpendicular (slope1 a) (slope2 a)) ∧
  ¬(are_perpendicular (slope1 a) (slope2 a) → a = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1233_123379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_product_l1233_123315

theorem arithmetic_sequence_max_product (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∃ d : ℝ, ∀ n, a (n + 1) = a n + d) →  -- Arithmetic sequence definition
  (a 3 + 2 * a 6 = 6) →  -- Given condition
  (∃ M : ℝ, (∀ d : ℝ, a 4 * a 6 ≤ M) ∧ (∃ d : ℝ, a 4 * a 6 = M)) ∧ 
  (∀ M' : ℝ, ((∀ d : ℝ, a 4 * a 6 ≤ M') ∧ (∃ d : ℝ, a 4 * a 6 = M')) → M' ≤ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_product_l1233_123315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_problem_l1233_123374

/-- Taxi fare structure -/
structure TaxiFare where
  initialPrice : ℚ
  chargePerKm : ℚ
  fuelSurcharge : ℚ

/-- Calculate fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℚ) : ℚ :=
  tf.initialPrice + max 0 (distance - 2) * tf.chargePerKm + tf.fuelSurcharge

theorem taxi_fare_problem (tf : TaxiFare) :
  calculateFare tf 3 = 9 ∧ 
  calculateFare tf 6 = 15 →
  tf.initialPrice = 6 ∧ 
  tf.chargePerKm = 2 ∧
  calculateFare tf 18 = 39 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_problem_l1233_123374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_equivalence_sum_equality_l1233_123309

/-- The general term of the original series -/
noncomputable def originalTerm (k : ℕ) : ℝ := 3^k / (9^k - 2)

/-- The general term of the telescoping series -/
noncomputable def telescopingTerm (k : ℕ) : ℝ := 1 / (3^k - 1) - 2 / (3^(2*k) - 2)

/-- Theorem stating that the original series can be rewritten as the telescoping series -/
theorem series_equivalence :
  ∀ k : ℕ, originalTerm k = telescopingTerm k := by
  sorry

/-- The infinite sum of the original series -/
noncomputable def originalSum : ℝ := ∑' k, originalTerm k

/-- The infinite sum of the telescoping series -/
noncomputable def telescopingSum : ℝ := ∑' k, telescopingTerm k

/-- Theorem stating that the sums of both series are equal -/
theorem sum_equality : originalSum = telescopingSum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_equivalence_sum_equality_l1233_123309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l1233_123373

noncomputable def cost_price : ℝ := 3300
noncomputable def selling_price : ℝ := 1230

noncomputable def loss_amount : ℝ := cost_price - selling_price

noncomputable def loss_percentage : ℝ := (loss_amount / cost_price) * 100

theorem loss_percentage_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |loss_percentage - 62.73| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l1233_123373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_sqrt_six_over_three_l1233_123353

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis length
  b : ℝ  -- Semi-minor axis length
  h₀ : a > 0
  h₁ : b > 0
  h₂ : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

theorem ellipse_eccentricity_is_sqrt_six_over_three :
  ∃ (e : Ellipse),
    e.a = 3 ∧                        -- One vertex is at (3, 0)
    2 * e.b^2 / e.a = 2 ∧            -- Chord length condition
    e.eccentricity = Real.sqrt 6 / 3 -- Eccentricity is √6/3
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_sqrt_six_over_three_l1233_123353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quasi_symmetric_point_of_f_l1233_123347

open Real

/-- Definition of the function f --/
noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4*log x

/-- Definition of the tangent line g at point x₀ --/
noncomputable def g (x₀ x : ℝ) : ℝ := (2*x₀ + 4/x₀ - 6)*(x - x₀) + x₀^2 - 6*x₀ + 4*log x₀

/-- Definition of quasi-symmetric point --/
def is_quasi_symmetric_point (x₀ : ℝ) : Prop :=
  ∀ x, x ≠ x₀ → (f x - g x₀ x) / (x - x₀) > 0

/-- Theorem: √2 is the unique quasi-symmetric point of f --/
theorem quasi_symmetric_point_of_f :
  ∃! x₀, x₀ > 0 ∧ is_quasi_symmetric_point x₀ ∧ x₀ = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quasi_symmetric_point_of_f_l1233_123347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_circles_intersection_l1233_123366

/-- The area of intersection of two circles with radii r₁ and r₂, 
    whose centers are separated by a distance d. -/
noncomputable def area_of_intersection (r₁ r₂ d : ℝ) : ℝ :=
  r₁^2 * Real.arccos ((d^2 + r₁^2 - r₂^2) / (2 * d * r₁)) +
  r₂^2 * Real.arccos ((d^2 + r₂^2 - r₁^2) / (2 * d * r₂)) -
  1/2 * Real.sqrt ((-d + r₁ + r₂) * (d + r₁ - r₂) * (d - r₁ + r₂) * (d + r₁ + r₂))

/-- Theorem stating that the area of intersection of two specific circles is correct. -/
theorem area_of_specific_circles_intersection :
  area_of_intersection 10 6 8 = area_of_intersection 10 6 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_circles_intersection_l1233_123366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1233_123370

/-- A sequence a is geometric if there exists a common ratio r such that
    a(n+1) = r * a(n) for all n. --/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometricSequence a) :
  a 4 * a 10 = 16 → a 7 = 4 ∨ a 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1233_123370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1233_123394

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from point P(1, 0) to the line x - y - 3 = 0 is √2 -/
theorem distance_point_to_line :
  distancePointToLine 1 0 1 (-1) (-3) = Real.sqrt 2 := by
  sorry

#check distance_point_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1233_123394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_without_consecutive_numbers_l1233_123329

/-- Counts the number of non-empty subsets of {1, ..., n} with no consecutive numbers -/
def count_subsets : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| n + 3 => count_subsets (n + 2) + count_subsets (n + 1) + 1

/-- The set of numbers we're considering -/
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}

theorem subsets_without_consecutive_numbers :
  count_subsets 10 = 143 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_without_consecutive_numbers_l1233_123329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_sum_l1233_123327

noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def originalRadius : ℝ := 12

noncomputable def largerVolume : ℝ := 3 * sphereVolume originalRadius

noncomputable def largerRadius : ℝ := originalRadius * Real.rpow 3 (1/3)

noncomputable def largerDiameter : ℝ := 2 * largerRadius

theorem sphere_diameter_sum :
  largerDiameter = 24 * Real.rpow 3 (1/3) ∧ 24 + 3 = 27 := by
  sorry

#eval 24 + 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_sum_l1233_123327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_T_l1233_123365

-- Define the points B and C
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (5, 0)

-- Define the area of triangle ABC
noncomputable def triangle_area (A : ℝ × ℝ) : ℝ :=
  abs ((A.1 - B.1) * (C.2 - B.2) - (C.1 - B.1) * (A.2 - B.2)) / 2

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {A | triangle_area A = 4}

-- Define a type for lines in ℝ²
def Line : Type := ℝ × ℝ → Prop

-- Define a parallel relation for lines
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem characterization_of_T :
  ∃ (l₁ l₂ : Line), 
    Parallel l₁ l₂ ∧ 
    l₁ (3, 2) ∧
    T = {p | l₁ p ∨ l₂ p} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_T_l1233_123365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_wheel_revolution_time_l1233_123352

/-- Represents the properties of a rotating wheel -/
structure Wheel where
  radius : ℝ
  revolutions_per_minute : ℝ

/-- Calculates the time for one revolution of a wheel in seconds -/
noncomputable def time_for_one_revolution (w : Wheel) : ℝ :=
  60 / w.revolutions_per_minute

/-- Given two engaging wheels with specific properties, proves that the larger wheel takes 6 seconds for one revolution -/
theorem larger_wheel_revolution_time
  (small_wheel large_wheel : Wheel)
  (h1 : large_wheel.radius = 3 * small_wheel.radius)
  (h2 : small_wheel.revolutions_per_minute = 30) :
  time_for_one_revolution large_wheel = 6 := by
  sorry

#check larger_wheel_revolution_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_wheel_revolution_time_l1233_123352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_periodic_l1233_123355

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | (n + 2) => (5 * sequence_a (n + 1) - 13) / (3 * sequence_a (n + 1) - 7)

theorem sequence_a_periodic : ∀ n : ℕ, n ≥ 1 → sequence_a (n + 3) = sequence_a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_periodic_l1233_123355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_2_2_l1233_123396

-- Define the function as noncomputable
noncomputable def f (a x : ℝ) : ℝ := a^(x-2) + (Real.log (x - 1)) / (Real.log a) + 1

-- State the theorem
theorem function_passes_through_point_2_2 (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  f a 2 = 2 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp [ha, ha']
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_2_2_l1233_123396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l1233_123392

-- Define the constants from the problem
noncomputable def freddy_time : ℝ := 4
noncomputable def eddy_distance : ℝ := 510
noncomputable def freddy_distance : ℝ := 300
noncomputable def speed_ratio : ℝ := 2.2666666666666666

-- Define Eddy's travel time as a function of the given parameters
noncomputable def eddy_time (ft : ℝ) (ed : ℝ) (fd : ℝ) (sr : ℝ) : ℝ :=
  ed / (sr * (fd / ft))

-- Theorem statement
theorem eddy_travel_time :
  eddy_time freddy_time eddy_distance freddy_distance speed_ratio = 3 := by
  -- Unfold the definition of eddy_time
  unfold eddy_time
  -- Perform the calculation
  norm_num
  -- Complete the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l1233_123392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_arc_measure_l1233_123360

/-- Two concentric circles with a point inside the smaller circle -/
structure ConcentricCircles where
  center : EuclideanSpace ℝ (Fin 2)
  inner_radius : ℝ
  outer_radius : ℝ
  inner_point : EuclideanSpace ℝ (Fin 2)
  h_inner_radius_pos : 0 < inner_radius
  h_outer_radius_gt_inner : inner_radius < outer_radius
  h_point_inside : ‖center - inner_point‖ < inner_radius

/-- An arc on a circle -/
structure CircleArc (c : ConcentricCircles) where
  start_point : EuclideanSpace ℝ (Fin 2)
  end_point : EuclideanSpace ℝ (Fin 2)
  h_on_circle : ‖c.center - start_point‖ = ‖c.center - end_point‖

/-- The angular measure of an arc -/
noncomputable def angular_measure (c : ConcentricCircles) (arc : CircleArc c) : ℝ := sorry

/-- The angle formed by two points and a vertex -/
noncomputable def angle_measure (vertex : EuclideanSpace ℝ (Fin 2)) (p1 : EuclideanSpace ℝ (Fin 2)) (p2 : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

theorem concentric_circles_arc_measure 
  (c : ConcentricCircles) 
  (α : ℝ)
  (outer_arc : CircleArc c)
  (inner_arc : CircleArc c)
  (h_outer_on_larger : ‖c.center - outer_arc.start_point‖ = c.outer_radius)
  (h_inner_on_smaller : ‖c.center - inner_arc.start_point‖ = c.inner_radius)
  (h_same_angle : angle_measure c.inner_point outer_arc.start_point outer_arc.end_point = α)
  (h_outer_measure : angular_measure c outer_arc = α) :
  angular_measure c inner_arc = α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_arc_measure_l1233_123360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_5y_l1233_123395

/-- The probability of selecting a point (x, y) from a rectangle with width 4040 and height 2020,
    such that x > 5y, is equal to 101/505. -/
theorem probability_x_gt_5y : 
  let rectangle_width : ℝ := 4040
  let rectangle_height : ℝ := 2020
  let total_area : ℝ := rectangle_width * rectangle_height
  let favorable_area : ℝ := (1/2) * rectangle_width * (rectangle_width / 5)
  favorable_area / total_area = 101 / 505 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_5y_l1233_123395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_logarithm_calculation_l1233_123376

theorem expression_evaluation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(2/3 : ℝ) * b^(-1 : ℝ))^(-1/2 : ℝ) * a^(1/2 : ℝ) * b^(1/3 : ℝ) / (a*b^5)^(1/6 : ℝ) = a^(1/6 : ℝ) :=
by sorry

theorem logarithm_calculation :
  1/2 * Real.log (32/49) - 4/3 * Real.log (Real.sqrt 8) + Real.log (Real.sqrt 245) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_logarithm_calculation_l1233_123376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_3_or_11_less_than_200_l1233_123313

theorem multiples_of_3_or_11_less_than_200 : 
  (Finset.filter (λ n : ℕ ↦ n < 200 ∧ ((n % 3 = 0 ∨ n % 11 = 0) ∧ ¬(n % 3 = 0 ∧ n % 11 = 0))) (Finset.range 200)).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_3_or_11_less_than_200_l1233_123313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sides_proof_l1233_123378

theorem dice_sides_proof :
  ∃ (n : ℕ), n > 0 ∧ 1 - (1 - 1 / (n : ℚ))^3 = 111328125 / 1000000000 ∧ n = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sides_proof_l1233_123378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l1233_123382

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the four propositions
theorem all_propositions_true 
  (m n : Line) (α β : Plane) :
  (parallel_lines m n ∧ line_in_plane m α ∧ plane_intersection α β n → parallel_lines m n) ∧
  (perpendicular_line_plane m α ∧ parallel_line_plane n β ∧ parallel_planes α β → perpendicular_lines m n) ∧
  (perpendicular_line_plane m α ∧ parallel_line_plane n β ∧ parallel_planes α β → perpendicular_lines m n) ∧
  (perpendicular_line_plane m α ∧ perpendicular_line_plane n β ∧ perpendicular_planes α β → perpendicular_lines m n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l1233_123382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_angle_is_60_degrees_l1233_123393

/-- Represents a folded rectangle as described in the problem -/
structure FoldedRectangle where
  /-- The original rectangle -/
  rect : Rectangle
  /-- The point where the corner is folded to on the centerline -/
  fold_point : Point

/-- The acute angle formed by the fold in a FoldedRectangle -/
noncomputable def acute_fold_angle (fr : FoldedRectangle) : ℝ :=
  sorry

/-- Theorem stating that the acute angle formed by the fold is 60° -/
theorem fold_angle_is_60_degrees (fr : FoldedRectangle) :
  acute_fold_angle fr = 60 := by
  sorry

#check fold_angle_is_60_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_angle_is_60_degrees_l1233_123393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_equality_implies_a_equals_one_l1233_123307

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2/a^2 - y^2/3 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the right focus of the hyperbola
noncomputable def hyperbola_right_focus (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + 3), 0)

-- Theorem statement
theorem focus_equality_implies_a_equals_one (a : ℝ) :
  a > 0 →
  parabola_focus = hyperbola_right_focus a →
  a = 1 :=
by
  intros h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_equality_implies_a_equals_one_l1233_123307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_and_magnitude_range_l1233_123367

noncomputable section

/-- Given two non-collinear vectors a and b in a real inner product space, 
    and points A, B, C defined by OA = a, OB = t*b, OC = (1/3)*(a + b),
    prove that A, B, and C are collinear when t = 1/2,
    and that for |a| = |b| = 1 with angle 120° between them,
    the range of |a - x*b| for x in [-1, 1/2] is [√3/2, √7/2]. -/
theorem vector_collinearity_and_magnitude_range 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (h_non_collinear : ¬ ∃ (r : ℝ), a = r • b) :
  (∃ (t : ℝ), ∃ (l : ℝ), (1/3 : ℝ) • (a + b) = l • a + (1 - l) • (t • b) ∧ t = 1/2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) (1/2 : ℝ) → 
    (‖a‖ = 1 ∧ ‖b‖ = 1 ∧ inner a b = -(1/2 : ℝ)) → 
      Real.sqrt (3/4 : ℝ) ≤ ‖a - x • b‖ ∧ ‖a - x • b‖ ≤ Real.sqrt (7/4 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_and_magnitude_range_l1233_123367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_inequality_l1233_123350

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.00010101 * (10 : ℝ)^m ≤ 10)) ∧ 
  (0.00010101 * (10 : ℝ)^k > 10) →
  k = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_inequality_l1233_123350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AMB_area_l1233_123337

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, 2)

/-- Point A: intersection of the original parabola with the y-axis -/
def point_A : ℝ × ℝ := (0, parabola 0)

/-- The rotated parabola equation -/
def rotated_parabola (x : ℝ) : ℝ := -(x^2 + 2*x - 1)

/-- Point B: intersection of the rotated parabola with the y-axis -/
def point_B : ℝ × ℝ := (0, rotated_parabola 0)

/-- The area of triangle AMB -/
noncomputable def triangle_area : ℝ := 
  (1/2) * |vertex.1 - point_A.1| * |point_A.2 - point_B.2|

theorem triangle_AMB_area :
  triangle_area = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AMB_area_l1233_123337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_nonagon_perimeter_l1233_123357

/-- A nonagon with specific angle measures and area -/
structure SpecialNonagon where
  /-- The area of the nonagon -/
  area : ℝ
  /-- The number of 90° angles in the nonagon -/
  right_angles : ℕ
  /-- The number of 60° angles in the nonagon -/
  sixty_degree_angles : ℕ
  /-- The sum of interior angles of a nonagon is (n-2) * 180° where n = 9 -/
  angle_sum : right_angles * 90 + sixty_degree_angles * 60 = 7 * 180

/-- The perimeter of a special nonagon -/
noncomputable def perimeter (n : SpecialNonagon) : ℝ :=
  18 * Real.sqrt 15 / 5

/-- Theorem stating the perimeter of a special nonagon with area 9√3 -/
theorem special_nonagon_perimeter (n : SpecialNonagon) 
  (h_area : n.area = 9 * Real.sqrt 3)
  (h_right : n.right_angles = 5)
  (h_sixty : n.sixty_degree_angles = 4) : 
  perimeter n = 18 * Real.sqrt 15 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_nonagon_perimeter_l1233_123357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_m_range_l1233_123324

/-- Given a curve f(x) = e^x - mx + 1 and a real number m, 
    if there exists a point where the tangent line of f is perpendicular to y = e^x, 
    then m > 1/e -/
theorem tangent_perpendicular_implies_m_range (m : ℝ) : 
  (∃ x : ℝ, (Real.exp x - m = -(Real.exp x)⁻¹)) → m > (Real.exp 1)⁻¹ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_m_range_l1233_123324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1233_123333

/-- Given a triangle ABC and a point M on the line AB, prove that if CM = -2CA + λCB, then λ = 3 -/
theorem triangle_vector_relation (A B C M : EuclideanSpace ℝ (Fin 2)) :
  (∃ t : ℝ, M = A + t • (B - A)) →
  (∃ l : ℝ, C - M = -2 • (C - A) + l • (C - B)) →
  ∃ l : ℝ, C - M = -2 • (C - A) + l • (C - B) ∧ l = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1233_123333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l1233_123351

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 20 cm, height 11 cm, and area 209 cm², 
    the other side is 18 cm -/
theorem trapezium_other_side (t : Trapezium) 
    (h1 : t.side1 = 20)
    (h2 : t.height = 11)
    (h3 : t.area = 209)
    (h4 : t.area = trapezium_area t) : 
  t.side2 = 18 := by
  sorry

#eval "Trapezium theorem defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l1233_123351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_solution_l1233_123390

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem largest_integer_solution :
  ∀ x : ℕ, x > 104 → floor (x / 7 : ℝ) ≠ floor (x / 8 : ℝ) + 1 :=
by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_solution_l1233_123390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_g_20_l1233_123334

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- g₁(n) is three times the number of positive integer divisors of n -/
def g₁ (n : ℕ) : ℕ := 3 * num_divisors n

/-- gⱼ(n) for j ≥ 0 -/
def g (j : ℕ) (n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | 1 => g₁ n
  | j+1 => g₁ (g j n)

theorem no_solution_for_g_20 :
  ∀ n : ℕ, n ≤ 40 → g 20 n ≠ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_g_20_l1233_123334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_point_configuration_theorem_l1233_123364

/-- A configuration of six points in the plane -/
structure SixPointConfig where
  points : Fin 6 → ℝ × ℝ

/-- Predicate to check if three points form a triangle -/
def forms_triangle (p q r : ℝ × ℝ) : Prop :=
  ¬ (∃ t : ℝ, (1 - t) • p + t • q = r)

/-- Predicate to check if two triangles are congruent -/
def congruent_triangles (p1 q1 r1 p2 q2 r2 : ℝ × ℝ) : Prop :=
  dist p1 q1 = dist p2 q2 ∧
  dist q1 r1 = dist q2 r2 ∧
  dist r1 p1 = dist r2 p2

/-- Predicate to check if points are in general position -/
def general_position (config : SixPointConfig) : Prop :=
  ∀ (i j k : Fin 6), i ≠ j → j ≠ k → i ≠ k →
    forms_triangle (config.points i) (config.points j) (config.points k)

/-- Predicate to check if the configuration satisfies the congruence condition -/
def satisfies_congruence_condition (config : SixPointConfig) : Prop :=
  ∀ (i j k l m n : Fin 6),
    i ≠ j → j ≠ k → k ≠ i →
    l ≠ m → m ≠ n → n ≠ l →
    i ≠ l → i ≠ m → i ≠ n → j ≠ l → j ≠ m → j ≠ n → k ≠ l → k ≠ m → k ≠ n →
    congruent_triangles
      (config.points i) (config.points j) (config.points k)
      (config.points l) (config.points m) (config.points n)

/-- Predicate to check if points form two equilateral triangles inscribed in a circle -/
def forms_two_equilateral_triangles_in_circle (config : SixPointConfig) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ i, dist center (config.points i) = radius) ∧
    (∃ (i j k l m n : Fin 6),
      i ≠ j → j ≠ k → k ≠ i →
      l ≠ m → m ≠ n → n ≠ l →
      i ≠ l → i ≠ m → i ≠ n → j ≠ l → j ≠ m → j ≠ n → k ≠ l → k ≠ m → k ≠ n →
      congruent_triangles
        (config.points i) (config.points j) (config.points k)
        (config.points i) (config.points j) (config.points k) ∧
      congruent_triangles
        (config.points l) (config.points m) (config.points n)
        (config.points l) (config.points m) (config.points n))

theorem six_point_configuration_theorem (config : SixPointConfig) :
  general_position config →
  satisfies_congruence_condition config →
  forms_two_equilateral_triangles_in_circle config :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_point_configuration_theorem_l1233_123364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1233_123311

/-- The curve C in the xy-plane -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2*p.1 = 0}

/-- The point M -/
def M : ℝ × ℝ := (0, 2)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_MN :
  (∀ N ∈ C, distance M N ≤ Real.sqrt 5 + 1) ∧
  (∃ N ∈ C, distance M N = Real.sqrt 5 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1233_123311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_cos_over_one_plus_sin_squared_l1233_123375

-- Define the function f as the indefinite integral
noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.log (1 + Real.sin x ^ 2)

-- State the theorem
theorem integral_sin_cos_over_one_plus_sin_squared (x : ℝ) :
  deriv f x = (Real.sin x * Real.cos x) / (1 + Real.sin x ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_cos_over_one_plus_sin_squared_l1233_123375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_bounds_l1233_123341

/-- Sequence S_n defined as 2^(2n) -/
def S (n : ℕ) : ℕ := 2^(2*n)

/-- Definition of the rectangle area given x and y coordinates -/
def rectangleArea (x y : List ℕ) : ℕ := (x.sum) * (y.sum)

/-- Theorem stating the maximum and minimum areas of the rectangle OABC -/
theorem rectangle_area_bounds (n : ℕ) (x y : List ℕ) 
  (h1 : x.length = n ∧ y.length = n)
  (h2 : ∀ k, 1 ≤ k ∧ k ≤ n → x.get? (k-1) = some (x[k-1]!) ∧ y.get? (k-1) = some (y[k-1]!) ∧ x[k-1]! * y[k-1]! = S k)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ n → x[k-1]! > 0 ∧ y[k-1]! > 0) :
  rectangleArea x y ≤ (1/3) * (4^n - 1) * (4^n + n - 1) ∧
  rectangleArea x y ≥ 4 * (2^n - 1)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_bounds_l1233_123341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_translation_period_min_omega_value_l1233_123330

open Real

theorem cos_translation_period (ω : ℝ) (h : ω > 0) :
  (∀ x : ℝ, Real.cos (ω * (x + π / 3)) = Real.cos (ω * x)) ↔ ∃ k : ℤ, ω = 6 * k :=
sorry

theorem min_omega_value :
  ∃! ω : ℝ, ω > 0 ∧ (∀ x : ℝ, Real.cos (ω * (x + π / 3)) = Real.cos (ω * x)) ∧
    ∀ ω' : ℝ, ω' > 0 → (∀ x : ℝ, Real.cos (ω' * (x + π / 3)) = Real.cos (ω' * x)) → ω ≤ ω' :=
sorry

#check min_omega_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_translation_period_min_omega_value_l1233_123330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_occupied_squares_l1233_123304

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a position on the chessboard -/
structure Position (n : ℕ) where
  row : Fin n
  col : Fin n

/-- Represents the color of a square on the chessboard -/
inductive Color
  | RedBlack
  | RedWhite
  | BlueBlack
  | BlueWhite

/-- Function to determine the color of a square based on its position -/
def squareColor (n : ℕ) (pos : Position n) : Color :=
  match (pos.row.val % 2, pos.col.val % 2) with
  | (0, 0) => Color.RedBlack
  | (0, 1) => Color.RedWhite
  | (1, 0) => Color.BlueBlack
  | (1, 1) => Color.BlueWhite
  | _ => Color.RedBlack  -- Default case to cover all possibilities

/-- Theorem stating the minimum number of always-occupied squares -/
theorem min_occupied_squares (n : ℕ) (board : Chessboard n) :
  ∃ (S : Finset (Position n)), S.card = 4 ∧
    ∀ (t : ℕ), ∃ (pos : Position n), pos ∈ S ∧
      ∀ (new_pos : Position n), (squareColor n new_pos = squareColor n pos) →
        new_pos ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_occupied_squares_l1233_123304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1233_123387

theorem problem_statement (x y : ℝ) (h1 : (2 : ℝ)^x = 36) (h2 : (3 : ℝ)^y = 36) : 
  (x * y = 2 * (x + y)) ∧ 
  (x * y > 16) ∧ 
  (x + y < 9) ∧ 
  (x^2 + y^2 > 32) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1233_123387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_numbers_theorem_l1233_123344

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Fin 10
  ones : Fin 10

/-- Sum of digits of a two-digit number -/
def digitSum (n : TwoDigitNumber) : Nat :=
  n.tens.val + n.ones.val

/-- Checks if two two-digit numbers are harmonic -/
def isHarmonic (a b : TwoDigitNumber) : Prop :=
  digitSum a = digitSum b

/-- Converts a two-digit number to its decimal representation -/
def toNatural (n : TwoDigitNumber) : Nat :=
  10 * n.tens.val + n.ones.val

theorem harmonic_numbers_theorem (a b : TwoDigitNumber) :
  a ≠ b →
  isHarmonic a b →
  toNatural a + toNatural b = 3 * (toNatural b - toNatural a) →
  toNatural a ∈ ({18, 27, 36, 45} : Finset Nat) := by
  sorry

#check harmonic_numbers_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_numbers_theorem_l1233_123344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_l1233_123318

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := (1/6) * a n * (a n + 3)

theorem arithmetic_sequence (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n a = (1/6) * a n * (a n + 3)) →
  (∃ k, ∀ n, a n = k * ↑n) :=
by
  intros h1 h2
  -- The proof goes here
  sorry

#check arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_l1233_123318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_our_parabola_l1233_123386

/-- A parabola is defined by its equation in the form y = a(x-h)^2 + k, where (h,k) is the vertex 
and 'a' determines the direction and width of the parabola. -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The focus of a parabola is a point on its axis of symmetry. -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (p.h, p.k + 1 / (4 * p.a))

/-- Our specific parabola y = (x-3)^2 -/
def our_parabola : Parabola := { a := 1, h := 3, k := 0 }

/-- Theorem: The focus of the parabola y = (x-3)^2 is at the point (3, 1/4) -/
theorem focus_of_our_parabola : focus our_parabola = (3, 1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_our_parabola_l1233_123386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1233_123342

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc 0 2 = Set.Icc (-3) 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1233_123342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seventeen_l1233_123381

theorem divisibility_by_seventeen (n : ℕ) :
  (17 ∣ (2^(4*n) - 1)) ∨ (17 ∣ (2^(4*n) + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seventeen_l1233_123381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l1233_123331

noncomputable def z : ℂ := (5 + Complex.I) / (1 + Complex.I)

theorem z_properties : 
  (z.im = -2) ∧ (z.re > 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l1233_123331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_theorem_l1233_123339

/-- Represents a rectangular box with given dimensions -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the triangle formed by center points of three faces -/
noncomputable def triangleArea (b : Box) : ℝ :=
  let side1 := Real.sqrt ((b.width / 2) ^ 2 + (b.length / 2) ^ 2)
  let side2 := Real.sqrt ((b.width / 2) ^ 2 + (b.height / 2) ^ 2)
  let side3 := Real.sqrt ((b.length / 2) ^ 2 + (b.height / 2) ^ 2)
  let s := (side1 + side2 + side3) / 2
  Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))

/-- Main theorem statement -/
theorem box_dimensions_theorem (p q : ℕ) (hpq : Nat.Coprime p q) :
  let b := Box.mk 15 20 (p / q : ℝ)
  triangleArea b = 50 → p + q = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_theorem_l1233_123339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_true_statements_l1233_123332

theorem exactly_three_true_statements : 
  let statement1 := ∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0)
  let statement2 := (∃ x : ℝ, x^2 + x + 1 = 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≠ 0)
  let statement3 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let statement4 := (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧ ¬(∀ x : ℝ, x^2 - 3*x + 2 > 0 → x > 2)
  statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_true_statements_l1233_123332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_satisfies_f_f_eq_5_l1233_123343

noncomputable def f (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then -x + 6
  else if -1 < x ∧ x ≤ 3 then x - 1
  else if 3 < x ∧ x ≤ 5 then -2*x + 10
  else 0  -- Default case for values outside the specified ranges

theorem unique_x_satisfies_f_f_eq_5 :
  ∃! x : ℝ, -4 ≤ x ∧ x ≤ 5 ∧ f (f x) = 5 := by
  sorry

#check unique_x_satisfies_f_f_eq_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_satisfies_f_f_eq_5_l1233_123343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l1233_123305

-- Define the ellipse and circle
def myEllipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def myCircle (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the conditions
variable (a b r : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : r > b)
variable (h4 : a > r)

-- Define the intersection points
def intersection_points (a b r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | myEllipse a b p.1 p.2 ∧ myCircle r p.1 p.2}

-- Define the area of the quadrilateral
noncomputable def quadrilateral_area (points : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem max_quadrilateral_area :
  ∃ (points : Set (ℝ × ℝ)), points = intersection_points a b r ∧
  ∀ (other_points : Set (ℝ × ℝ)), other_points = intersection_points a b r →
  quadrilateral_area points ≥ quadrilateral_area other_points ∧
  quadrilateral_area points = 2 * a * b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l1233_123305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1233_123399

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (20, 0)

-- Define the initial slopes of the lines
def slope_ℓA : ℝ := 1
noncomputable def slope_ℓB : ℝ := Real.arctan (Real.pi / 2)  -- Approximation for vertical line
def slope_ℓC : ℝ := -1

-- Define the rotation angle
variable (α : ℝ)

-- Define the rotating lines (implicitly)
def ℓA (α : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = t * Real.cos α - t * Real.sin α ∧ p.2 = t * Real.sin α + t * Real.cos α}
def ℓB (α : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 12 - t * Real.sin α ∧ p.2 = t * Real.cos α}
def ℓC (α : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 20 + t * Real.cos α + t * Real.sin α ∧ p.2 = t * Real.sin α - t * Real.cos α}

-- Define the triangle area function
noncomputable def triangle_area (α : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area :
  ∃ (α : ℝ), ∀ (β : ℝ), triangle_area α ≥ triangle_area β ∧ triangle_area α = 104 := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1233_123399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_power_theorem_l1233_123371

-- Define the circle and points
variable (C : Set (ℝ × ℝ)) -- Circle C in 2D plane
variable (Q R X Y : ℝ × ℝ) -- Points in 2D plane

-- Define the conditions
variable (h1 : Q ∉ C) -- Q is outside circle C
variable (h2 : IsTangentLine C Q R) -- Line QR is tangent to C at R
variable (h3 : IsSecantLine C Q X Y) -- Line QXY is a secant of C
variable (h4 : dist Q X < dist Q Y) -- QX < QY
variable (h5 : dist Q X = 5) -- QX = 5
variable (h6 : dist Q R = dist X Y - dist Q X) -- QR = XY - QX

-- Theorem statement
theorem point_power_theorem : dist Q Y = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_power_theorem_l1233_123371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_line_equation_l1233_123361

/-- Given two points A and B that are symmetric about a line L, prove that the equation of L is x - y + 1 = 0 --/
theorem symmetric_points_line_equation (a : ℝ) :
  let A : ℝ × ℝ := (a - 1, a + 1)
  let B : ℝ × ℝ := (a, a)
  ∃ (L : Set (ℝ × ℝ)), (∀ (p : ℝ × ℝ), p ∈ L ↔ p.1 - p.2 + 1 = 0) ∧
    (∀ (p : ℝ × ℝ), p ∈ L → dist A p = dist B p) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_line_equation_l1233_123361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1233_123316

def a (l : ℝ) : ℝ × ℝ := (l, 2)
def b : ℝ × ℝ := (3, 4)

def angle_acute (v w : ℝ × ℝ) : Prop :=
  0 < v.1 * w.1 + v.2 * w.2 ∧ v.1 * w.1 + v.2 * w.2 < Real.sqrt ((v.1^2 + v.2^2) * (w.1^2 + w.2^2))

theorem lambda_range (l : ℝ) : 
  angle_acute (a l) b → l > -8/3 ∧ l ≠ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1233_123316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_l1233_123314

theorem new_average_age 
  (initial_people : ℕ) 
  (initial_average : ℚ) 
  (leaving_age : ℕ) 
  (entering_age : ℕ) 
  (h1 : initial_people = 7)
  (h2 : initial_average = 28)
  (h3 : leaving_age = 22)
  (h4 : entering_age = 30)
  : (initial_people * initial_average - leaving_age + entering_age) / initial_people = 29 + 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_age_l1233_123314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_and_intersection_l1233_123348

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 - (3/2) * x^2 + (a + 1) * x + 5

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + (a + 1)

noncomputable def g (x m : ℝ) : ℝ := (1/3) * x^3 - (3/2) * x^2 + 5 - m

noncomputable def g_derivative (x : ℝ) : ℝ := x^2 - 3 * x

theorem extreme_point_and_intersection (a : ℝ) :
  (∀ x, f_derivative a x = 0 → x = 1) →
  (∃ m, ∀ x, g x m = 0 → (x < 0 ∨ (0 < x ∧ x < 3) ∨ x > 3)) →
  (f a = f 1 ∧ ∀ m, (1/2 < m ∧ m < 5) ↔ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_and_intersection_l1233_123348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mural_cost_l1233_123358

/-- Calculate the total cost of painting a mural -/
theorem mural_cost (length width : ℝ) (paint_cost_per_sqm : ℝ) (artist_rate_sqm_per_hour : ℝ) (artist_hourly_rate : ℝ) :
  length = 6 →
  width = 3 →
  paint_cost_per_sqm = 4 →
  artist_rate_sqm_per_hour = 1.5 →
  artist_hourly_rate = 10 →
  (length * width * paint_cost_per_sqm) + ((length * width / artist_rate_sqm_per_hour) * artist_hourly_rate) = 192 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mural_cost_l1233_123358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approximately_34_16_l1233_123398

/-- Represents a section of the car's journey -/
structure Section where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a section -/
noncomputable def sectionTime (s : Section) : ℝ := s.distance / s.speed

/-- Calculates the average speed given a list of sections -/
noncomputable def averageSpeed (sections : List Section) : ℝ :=
  let totalDistance := sections.foldl (fun acc s => acc + s.distance) 0
  let totalTime := sections.foldl (fun acc s => acc + sectionTime s) 0
  totalDistance / totalTime

/-- The sections of the car's journey -/
def journey : List Section := [
  ⟨120, 30⟩, ⟨60, 40⟩, ⟨90, 35⟩, 
  ⟨80, 50⟩, ⟨100, 25⟩, ⟨70, 45⟩
]

theorem average_speed_approximately_34_16 : 
  abs (averageSpeed journey - 34.16) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approximately_34_16_l1233_123398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_monotonic_condition_l1233_123363

noncomputable section

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := (x + b) * Real.log x

-- Define the function g
def g (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := Real.exp x * (f b x / (x + 2) - 2 * a)

-- Theorem for part 1
theorem tangent_line_parallel (b : ℝ) :
  (deriv (f b) 1 = 3) → b = 2 :=
sorry

-- Theorem for part 2
theorem monotonic_condition (a : ℝ) (b : ℝ) :
  a ≠ 0 →
  (Monotone (g a b) ∨ StrictMono (g a b)) →
  a ≤ 1 / 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_monotonic_condition_l1233_123363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PRS_is_90_degrees_l1233_123389

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  scalene : P ≠ Q ∧ Q ≠ R ∧ R ≠ P
  right_angle_at_Q : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem angle_PRS_is_90_degrees 
  (P Q R : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (h_PR : (P.1 - R.1)^2 + (P.2 - R.2)^2 = 15^2)
  (h_circle : Q ∈ Circle P 15 ∧ R ∈ Circle P 15)
  (S : ℝ × ℝ)
  (h_S : S ∈ Circle P 15 ∧ ∃ t > 1, S.1 = Q.1 + t * (Q.1 - P.1) ∧ S.2 = Q.2 + t * (Q.2 - P.2))
  : ∃ (angle_PRS : ℝ), angle_PRS = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_PRS_is_90_degrees_l1233_123389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_minimum_l1233_123349

/-- The minimum value of f(x) = (x - 1)^2 + 1 on the interval [t, t+1] --/
noncomputable def min_value (t : ℝ) : ℝ :=
  if t > 1 then (t - 1)^2 + 1
  else if 0 ≤ t ∧ t ≤ 1 then 1
  else t^2 + 1

/-- The function f(x) = (x - 1)^2 + 1 --/
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem min_value_is_minimum (t : ℝ) :
  ∀ x ∈ Set.Icc t (t + 1), min_value t ≤ f x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_minimum_l1233_123349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l1233_123380

/-- A triangle in three-dimensional space -/
structure Triangle where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ

/-- A plane in three-dimensional space -/
structure Plane where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (point : ℝ × ℝ × ℝ) (plane : Plane) : Prop :=
  let (x, y, z) := point
  let (px, py, pz) := plane.point
  let (nx, ny, nz) := plane.normal
  nx * (x - px) + ny * (y - py) + nz * (z - pz) = 0

/-- Theorem: A triangle determines a unique plane in three-dimensional space -/
theorem triangle_determines_plane (t : Triangle) : 
  ∃! p : Plane, point_on_plane t.p1 p ∧ point_on_plane t.p2 p ∧ point_on_plane t.p3 p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l1233_123380
