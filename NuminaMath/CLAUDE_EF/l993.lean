import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l993_99366

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 1)
  (h2 : Real.cos t.B * Real.sin t.C - (t.a - Real.sin t.B) * Real.cos t.C = 0) :
  t.C = π/4 ∧ -1/2 ≤ t.a * t.b ∧ t.a * t.b ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l993_99366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_with_noncollinearity_l993_99365

/-- A bijection from natural numbers to the two-dimensional integer lattice -/
noncomputable def f : ℕ → ℤ × ℤ := sorry

/-- Collinearity of three points in ℤ² -/
def collinear (p q r : ℤ × ℤ) : Prop :=
  ∃ (a b c : ℤ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧
    a * (q.1 - p.1) + b * (r.1 - p.1) = 0 ∧
    a * (q.2 - p.2) + b * (r.2 - p.2) = 0

/-- The main theorem -/
theorem bijection_with_noncollinearity :
  Function.Bijective f ∧
  ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c →
    Nat.gcd a (Nat.gcd b c) > 1 →
    ¬collinear (f a) (f b) (f c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_with_noncollinearity_l993_99365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_score_for_mean_93_l993_99396

noncomputable def quiz_scores : List ℚ := [92, 95, 87, 89, 100]

def arithmetic_mean (scores : List ℚ) : ℚ :=
  scores.sum / scores.length

theorem sixth_score_for_mean_93 :
  let scores_with_sixth := quiz_scores ++ [95]
  arithmetic_mean scores_with_sixth = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_score_for_mean_93_l993_99396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_to_ben_l993_99395

/-- Represents the number of boys in the circle -/
def n : ℕ := 14

/-- Represents the number of boys skipped in each throw plus one (the receiver) -/
def k : ℕ := 5

/-- The sequence of boys who receive the ball -/
def ballSequence : ℕ → ℕ
  | 0 => 1  -- Start with Ben (boy number 1)
  | i + 1 => ((ballSequence i + k - 1) % n) + 1

/-- The number of throws needed for the ball to return to Ben -/
def numThrows : ℕ := (List.range n).findIdx? (fun i => ballSequence i = 1) |>.getD n

theorem ball_returns_to_ben : numThrows = 13 := by
  sorry

#eval numThrows

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_to_ben_l993_99395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option1_better_than_option2_l993_99340

-- Define the probabilities
noncomputable def p_traditional : ℝ := 1 / 2
noncomputable def p_new : ℝ := 1 / 3

-- Define the scoring rules
def score_traditional (completed : ℕ) : ℝ := 30 * completed
def score_new (completed : ℕ) : ℝ :=
  match completed with
  | 0 => 0
  | 1 => 40
  | _ => 90

-- Define the expected score for Option 1
noncomputable def expected_score_option1 : ℝ :=
  3 * p_traditional * score_traditional 1

-- Define the expected score for Option 2
noncomputable def expected_score_option2 : ℝ :=
  p_traditional * score_traditional 1 +
  (p_new * (1 - p_new) * score_new 1) +
  (p_new * p_new * score_new 2)

-- Theorem to prove
theorem option1_better_than_option2 :
  expected_score_option1 > expected_score_option2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option1_better_than_option2_l993_99340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_complex_plane_l993_99389

/-- Given complex numbers and their corresponding points on the complex plane,
    prove that lambda + mu = 1 when OC = lambda*OA + mu*OB --/
theorem vector_equation_complex_plane (z₁ z₂ z₃ : ℂ) (A B C : ℂ) (lambda mu : ℝ) :
  z₁ = -1 + 2*I →
  z₂ = 1 - I →
  z₃ = 3 - 4*I →
  A = z₁ →
  B = z₂ →
  C = z₃ →
  C = lambda • A + mu • B →
  lambda + mu = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_complex_plane_l993_99389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_approx_42_55_l993_99326

-- Define the train's length in meters
noncomputable def train_length : ℝ := 780

-- Define the train's speed in km/h
noncomputable def train_speed : ℝ := 60

-- Define the man's speed in km/h
noncomputable def man_speed : ℝ := 6

-- Define the relative speed in km/h
noncomputable def relative_speed : ℝ := train_speed + man_speed

-- Define the conversion factor from km/h to m/s
noncomputable def km_h_to_m_s : ℝ := 1000 / 3600

-- Define the relative speed in m/s
noncomputable def relative_speed_m_s : ℝ := relative_speed * km_h_to_m_s

-- Define the time to cross in seconds
noncomputable def time_to_cross : ℝ := train_length / relative_speed_m_s

-- Theorem stating that the time to cross is approximately 42.55 seconds
theorem time_to_cross_approx_42_55 :
  ∃ ε > 0, abs (time_to_cross - 42.55) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_approx_42_55_l993_99326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_g_l993_99369

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 2)

-- Define g as a linear function
noncomputable def g (m c : ℝ) (x : ℝ) : ℝ := m * x + c

-- State the theorem
theorem x_intercept_of_g (m c : ℝ) :
  (∃ (g_inv : ℝ → ℝ), Function.LeftInverse g_inv (g m c) ∧ Function.RightInverse g_inv (g m c)) →
  (∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f) →
  (∀ f_inv, Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f → f_inv (g m c 2) = 7) →
  (∀ g_inv, Function.LeftInverse g_inv (g m c) ∧ Function.RightInverse g_inv (g m c) → g_inv (f 1) = 4/5) →
  (g m c 0 = 0 → 7/5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_g_l993_99369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_negative_positive_negative_l993_99368

theorem simplify_negative_positive_negative : -(-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_negative_positive_negative_l993_99368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l993_99343

theorem all_propositions_true :
  (∀ x : ℝ, x^2 + 2*x + 7 > 0) ∧
  (∃ x : ℝ, x + 1 > 0) ∧
  (∀ p q : Prop, (¬p → ¬q) → (q → p)) ∧
  (let p : Prop := ∀ S : Set ℕ, ∅ ⊆ S;
   let q : Prop := (0 : ℕ) ∈ (∅ : Set ℕ);
   (p ∨ q) ∧ ¬(p ∧ q)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l993_99343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l993_99377

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 + 1

/-- Point C -/
def C : ℝ × ℝ := (1, 2)

/-- Distance squared between two points -/
noncomputable def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Sum of reciprocals of squared distances -/
noncomputable def sumReciprocalDistances (A B : ℝ × ℝ) : ℝ :=
  1 / distanceSquared C A + 1 / distanceSquared C B

/-- Theorem statement -/
theorem chord_reciprocal_sum_constant :
  ∀ (A B : ℝ × ℝ),
  A.2 = parabola A.1 → B.2 = parabola B.1 →
  (B.2 - A.2) / (B.1 - A.1) = (C.2 - A.2) / (C.1 - A.1) →
  sumReciprocalDistances A B = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l993_99377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l993_99367

/-- The equation we want to solve -/
noncomputable def f (x : ℝ) : ℝ :=
  8 / (Real.sqrt (x - 10) - 10) + 2 / (Real.sqrt (x - 10) - 5) +
  9 / (Real.sqrt (x - 10) + 5) + 16 / (Real.sqrt (x - 10) + 10)

/-- There exists a unique solution to the equation f(x) = 0 close to 80.3329 -/
theorem solution_exists : ∃! x : ℝ, f x = 0 ∧ |x - 80.3329| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l993_99367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_longest_side_correct_l993_99319

/-- An isosceles triangle with base angle α and leg length a -/
structure IsoscelesTriangle where
  α : ℝ
  a : ℝ
  h : ℝ
  h_def : h = a * Real.sin α

/-- An inscribed triangle in the isosceles triangle -/
structure InscribedTriangle (T : IsoscelesTriangle) where
  longest_side : ℝ

/-- The minimum value of the longest side of any inscribed triangle -/
noncomputable def min_longest_side (T : IsoscelesTriangle) : ℝ :=
  if T.α ≥ Real.pi / 3
  then T.a / 2 * Real.cos T.α
  else 2 * T.a * T.h / (T.a * Real.sqrt 3 + 2 * T.h)

/-- Theorem stating that the minimum longest side is correct -/
theorem min_longest_side_correct (T : IsoscelesTriangle) :
  ∀ (D : InscribedTriangle T), D.longest_side ≥ min_longest_side T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_longest_side_correct_l993_99319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oatmeal_raisin_cookies_count_l993_99374

/-- The number of dozens of a specific baked good -/
structure Dozens where
  value : ℕ

/-- The price of a single item in dollars -/
structure Price where
  value : ℚ

/-- The total amount raised in dollars -/
structure TotalRaised where
  value : ℚ

/-- Calculate the number of items given the number of dozens -/
def itemCount (d : Dozens) : ℕ := d.value * 12

/-- Calculate the amount raised from a specific item -/
def amountRaised (d : Dozens) (p : Price) : ℚ := d.value * 12 * p.value

instance : OfNat Dozens n where
  ofNat := ⟨n⟩

instance : OfNat Price n where
  ofNat := ⟨n⟩

instance : OfNat TotalRaised n where
  ofNat := ⟨n⟩

theorem oatmeal_raisin_cookies_count 
  (betty_choc_chip : Dozens)
  (betty_reg_brownies : Dozens)
  (paige_sugar : Dozens)
  (paige_blondies : Dozens)
  (paige_cream_cheese : Dozens)
  (cookie_price : Price)
  (brownie_price : Price)
  (total_raised : TotalRaised)
  (h1 : betty_choc_chip = 4)
  (h2 : betty_reg_brownies = 2)
  (h3 : paige_sugar = 6)
  (h4 : paige_blondies = 3)
  (h5 : paige_cream_cheese = 5)
  (h6 : cookie_price = 1)
  (h7 : brownie_price = 2)
  (h8 : total_raised = 432)
  (h9 : total_raised.value = amountRaised betty_choc_chip cookie_price + 
                       amountRaised betty_reg_brownies brownie_price +
                       amountRaised paige_sugar cookie_price +
                       amountRaised paige_blondies brownie_price +
                       amountRaised paige_cream_cheese brownie_price +
                       amountRaised (betty_oatmeal_raisin) cookie_price) :
  betty_oatmeal_raisin = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oatmeal_raisin_cookies_count_l993_99374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_lambda_eq_three_fifths_l993_99361

def vec_a (l : ℝ) : Fin 2 → ℝ := ![3, l]
def vec_b (l : ℝ) : Fin 2 → ℝ := ![l - 1, 2]

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem perpendicular_iff_lambda_eq_three_fifths (l : ℝ) :
  perpendicular (vec_a l) (vec_b l) ↔ l = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_lambda_eq_three_fifths_l993_99361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l993_99386

open Real

/-- The function f(x) = ln x + x^2 - 1 -/
noncomputable def f (x : ℝ) : ℝ := log x + x^2 - 1

/-- The function g(x) = e^x - e -/
noncomputable def g (x : ℝ) : ℝ := exp x - exp 1

theorem inequality_equivalence (m : ℝ) :
  (∀ x > 1, m * g x > f x) ↔ m ≥ 3 / exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l993_99386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l993_99318

-- Define the ages as natural numbers
def patrick_age : ℕ → ℕ := sorry
def michael_age : ℕ → ℕ := sorry
def monica_age : ℕ → ℕ := sorry

-- Define the conditions
axiom ratio_patrick_michael : ∀ x : ℕ, patrick_age x * 5 = michael_age x * 3
axiom ratio_michael_monica : ∀ x : ℕ, michael_age x * 5 = monica_age x * 3
axiom sum_of_ages : ∀ x : ℕ, patrick_age x + michael_age x + monica_age x = 146

-- State the theorem
theorem age_difference : ∃ x : ℕ, monica_age x - patrick_age x = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l993_99318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_point_three_zeros_condition_l993_99316

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*x^2 - 4*x + 4)*Real.exp x - a*x^2 - Real.exp 1

noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ :=
  f a 1 + (2*Real.exp 1 - 2*a) * (x - 1)

theorem tangent_passes_through_point (a : ℝ) :
  tangent_line a 0 = 1 - Real.exp 1 → a = 1 := by sorry

theorem three_zeros_condition (a : ℝ) :
  a > 0 → (∃ x y z : ℝ, x < y ∧ y < z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  (∀ w : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z → f a w ≠ 0) →
  a > Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_point_three_zeros_condition_l993_99316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_implies_expression_value_l993_99335

theorem tan_value_implies_expression_value (x : ℝ) (h : Real.tan x = 1/2) :
  (3 * (Real.sin x)^2 - 2) / (Real.sin x * Real.cos x) = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_implies_expression_value_l993_99335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_eleven_meters_rebar_l993_99333

/-- The weight of rebar in kilograms for a given length in meters -/
noncomputable def rebarWeight (length : ℝ) : ℝ :=
  (15.3 / 5) * length

/-- Theorem stating that 11 meters of rebar weighs 33.66 kilograms -/
theorem weight_of_eleven_meters_rebar :
  rebarWeight 11 = 33.66 := by
  -- Unfold the definition of rebarWeight
  unfold rebarWeight
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_eleven_meters_rebar_l993_99333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l993_99328

theorem find_d : ∃ d : ℝ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * (n : ℝ)^2 + 11 * (n : ℝ) - 20 = 0) ∧
  (let frac := d - ⌊d⌋;
   4 * frac^2 - 12 * frac + 5 = 0 ∧ 
   0 ≤ frac ∧ frac < 1) ∧
  d = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l993_99328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fleas_cannot_align_l993_99387

/-- Represents a flea's position on a 2D plane -/
structure Flea where
  x : ℚ
  y : ℚ

/-- Represents the state of all four fleas -/
structure FleaState where
  fleas : Fin 4 → Flea

/-- Defines a valid initial state where fleas are at corners of a unit square -/
def validInitialState (state : FleaState) : Prop :=
  (state.fleas 0).x = 0 ∧ (state.fleas 0).y = 0 ∧
  (state.fleas 1).x = 1 ∧ (state.fleas 1).y = 0 ∧
  (state.fleas 2).x = 0 ∧ (state.fleas 2).y = 1 ∧
  (state.fleas 3).x = 1 ∧ (state.fleas 3).y = 1

/-- Defines a valid jump according to the rules -/
def validJump (before after : FleaState) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧
    (∀ k : Fin 4, k ≠ i → after.fleas k = before.fleas k) ∧
    let dx := (before.fleas j).x - (before.fleas i).x
    let dy := (before.fleas j).y - (before.fleas i).y
    (after.fleas i).x = (before.fleas j).x + dx ∧
    (after.fleas i).y = (before.fleas j).y + dy

/-- Checks if all fleas are on the same line -/
def allOnSameLine (state : FleaState) : Prop :=
  ∃ (a b : ℚ), ∀ i : Fin 4,
    a * (state.fleas i).x + b * (state.fleas i).y = 1 ∨
    a * (state.fleas i).x + b * (state.fleas i).y = 0

/-- Main theorem: It's impossible for fleas to align on a straight line after a series of jumps -/
theorem fleas_cannot_align :
  ¬∃ (initial final : FleaState) (n : ℕ) (states : Fin (n + 1) → FleaState),
    validInitialState initial ∧
    states 0 = initial ∧
    states n = final ∧
    (∀ i : Fin n, validJump (states i) (states (i.succ))) ∧
    allOnSameLine final :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fleas_cannot_align_l993_99387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_powers_of_two_l993_99385

theorem perfect_cubes_between_powers_of_two : 
  (Finset.filter (fun n => n^3 ≥ 2^8 + 1 ∧ n^3 ≤ 2^10 + 1) (Finset.range 11)).card = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_powers_of_two_l993_99385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l993_99337

/-- The line l: x - 2y - 20 = 0 -/
def line_l (x y : ℝ) : Prop := x - 2*y - 20 = 0

/-- The ellipse: x²/16 + y²/9 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- Point P on line l -/
def point_on_line_l (x y : ℝ) : Prop := line_l x y

/-- Tangent point on the ellipse -/
def tangent_point (x y : ℝ) : Prop := ellipse x y

/-- The fixed point Q -/
noncomputable def point_Q : ℝ × ℝ := (4/5, -9/10)

/-- Theorem stating that MN always passes through Q and Q bisects MN when MN is parallel to l -/
theorem fixed_point_theorem (x_p y_p x_m y_m x_n y_n : ℝ) 
  (h_p : point_on_line_l x_p y_p) 
  (h_m : tangent_point x_m y_m) 
  (h_n : tangent_point x_n y_n) :
  (∃ t : ℝ, x_m + t * (x_n - x_m) = (point_Q.1) ∧ 
            y_m + t * (y_n - y_m) = (point_Q.2)) ∧ 
  ((x_n - x_m) / (y_n - y_m) = 1/2 → 
    point_Q = ((x_m + x_n) / 2, (y_m + y_n) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l993_99337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_numbers_l993_99313

theorem no_valid_numbers : ¬∃ n : ℕ,
  (n ≥ 100000 ∧ n < 1000000) ∧  -- 6-digit number
  (∀ d ∈ n.digits 10, 2 ≤ d ∧ d ≤ 8) ∧  -- digits between 2 and 8
  (∀ d ∈ n.digits 10, d ≠ 0 ∧ d ≠ 1) ∧  -- no 0 or 1
  (n.digits 10).length = 6 ∧  -- 6 digits
  (n.digits 10).sum = 42 ∧  -- sum of digits is 42
  (n.digits 10).Nodup  -- no repeated digits
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_numbers_l993_99313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_integer_sin_l993_99322

/-- Definition of IsInt for Real numbers -/
def IsInt (x : Real) : Prop := ∃ n : Int, x = n

/-- Given b = π/1806, prove that 903 is the smallest positive integer m 
    such that sin(m(m+1)π/1806) is an integer. -/
theorem smallest_m_for_integer_sin (b : ℝ) (h : b = Real.pi / 1806) : 
  (∃ m : ℕ, m > 0 ∧ 
   (∀ k : ℕ, 0 < k → k < m → ¬ IsInt (Real.sin (k * (k + 1) * b))) ∧ 
   IsInt (Real.sin (m * (m + 1) * b))) ∧
  (∃ m : ℕ, m > 0 ∧ 
   (∀ k : ℕ, 0 < k → k < m → ¬ IsInt (Real.sin (k * (k + 1) * b))) ∧ 
   IsInt (Real.sin (m * (m + 1) * b)) → m = 903) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_integer_sin_l993_99322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l993_99312

/-- A coloring scheme for a cube -/
def ColoringScheme := Fin 6 → Fin 6

/-- Two coloring schemes are equivalent if they can be made to coincide through rotation -/
def equivalent_schemes (s1 s2 : ColoringScheme) : Prop := sorry

/-- A valid coloring scheme satisfies the adjacent face condition -/
def valid_scheme (s : ColoringScheme) : Prop := sorry

/-- The set of all valid coloring schemes -/
def valid_schemes : Set ColoringScheme := {s | valid_scheme s}

/-- The setoid for equivalent coloring schemes -/
def ColoringSchemeSetoid : Setoid ColoringScheme :=
{ r := equivalent_schemes,
  iseqv := sorry }

/-- The set of distinct coloring schemes up to rotational symmetry -/
def distinct_schemes : Set (Quotient ColoringSchemeSetoid) := sorry

/-- Assume that distinct_schemes is finite -/
instance : Fintype (Quotient ColoringSchemeSetoid) := sorry

theorem cube_coloring_count :
  Fintype.card (Quotient ColoringSchemeSetoid) = 230 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l993_99312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l993_99320

/-- A tetrahedron with two opposite edges of lengths a and b, and all other edges equal. -/
structure SpecialTetrahedron (a b : ℝ) where
  (a_positive : 0 < a)
  (b_positive : 0 < b)

/-- The radius of the inscribed sphere in a special tetrahedron. -/
noncomputable def inscribed_sphere_radius (t : SpecialTetrahedron a b) : ℝ :=
  (Real.sqrt (2 * a * b)) / 4

/-- Theorem: The radius of a sphere touching all edges of a special tetrahedron
    with opposite edges of lengths a and b is (√(2ab))/4. -/
theorem inscribed_sphere_radius_formula (a b : ℝ) (t : SpecialTetrahedron a b) :
  inscribed_sphere_radius t = (Real.sqrt (2 * a * b)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l993_99320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_width_is_55_l993_99338

/-- Represents the dimensions and cost of a rectangular plot with a gravel path -/
structure PlotWithPath where
  length : ℝ
  pathWidth : ℝ
  gravelCost : ℝ
  gravelPricePerSqm : ℝ

/-- Calculates the width of a rectangular plot given its dimensions and gravel path cost -/
noncomputable def calculatePlotWidth (plot : PlotWithPath) : ℝ :=
  let totalPathArea := plot.gravelCost / (plot.gravelPricePerSqm / 100)
  let outerLength := plot.length + 2 * plot.pathWidth
  ((totalPathArea + plot.length * plot.pathWidth) / outerLength) - plot.pathWidth

/-- Theorem stating that for the given plot specifications, the width is 55 meters -/
theorem plot_width_is_55 (plot : PlotWithPath) 
  (h1 : plot.length = 110)
  (h2 : plot.pathWidth = 2.5)
  (h3 : plot.gravelCost = 340)
  (h4 : plot.gravelPricePerSqm = 40) :
  calculatePlotWidth plot = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_width_is_55_l993_99338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_weight_problem_l993_99392

noncomputable def average_weight (total_weight : ℝ) (num_students : ℕ) : ℝ :=
  total_weight / num_students

noncomputable def total_weight (avg_weight : ℝ) (num_students : ℕ) : ℝ :=
  avg_weight * num_students

theorem class_weight_problem (section_A_students : ℕ) (section_B_students : ℕ) 
  (section_B_avg_weight : ℝ) (class_avg_weight : ℝ) :
  section_A_students = 50 →
  section_B_students = 70 →
  section_B_avg_weight = 70 →
  class_avg_weight = 61.67 →
  ∃ (section_A_avg_weight : ℝ), 
    (average_weight 
      (total_weight section_A_avg_weight section_A_students + 
       total_weight section_B_avg_weight section_B_students)
      (section_A_students + section_B_students) = class_avg_weight) ∧
    (abs (section_A_avg_weight - 50.01) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_weight_problem_l993_99392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l993_99341

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^5 + 3 = (X - 1)^2 * q + (5*X - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l993_99341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_f_l993_99373

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 3) + Real.pi / 6)

-- State the theorem
theorem symmetric_center_of_f : 
  ∃ (k : ℤ), f (Real.pi / 4 + k * Real.pi / 2) = 0 ∧ 
  (∀ (x : ℝ), f (Real.pi / 4 + k * Real.pi / 2 + x) = f (Real.pi / 4 + k * Real.pi / 2 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_f_l993_99373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l993_99350

/-- The x-coordinate of a point P on a hyperbola and a circle -/
theorem hyperbola_circle_intersection (b c : ℝ) (P : ℝ × ℝ) : 
  let (x, y) := P
  (x > 0) ∧ (y > 0) ∧  -- P is in the first quadrant
  (x^2 - y^2 / b^2 = 1) ∧  -- P is on the hyperbola
  (x^2 + y^2 = c^2) ∧  -- P is on the circle
  ((x - c)^2 + y^2 = (c + 2)^2)  -- |PF1| = c + 2
  →
  x = (Real.sqrt 3 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l993_99350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l993_99391

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x - a * Real.log x

-- State the theorem
theorem f_lower_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ 2 * a - (1/2) * a^2) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l993_99391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_matches_count_l993_99308

theorem initial_matches_count (initial_average : ℚ) (additional_runs : ℚ) (average_increase : ℚ) : 
  initial_average = 34 →
  additional_runs = 89 →
  average_increase = 5 →
  (∃ x : ℕ, x > 0 ∧ 
    (initial_average * x + additional_runs) / (x + 1) = initial_average + average_increase) →
  (∃ x : ℕ, x > 0 ∧ 
    (initial_average * x + additional_runs) / (x + 1) = initial_average + average_increase ∧ 
    x = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_matches_count_l993_99308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_properties_l993_99302

theorem square_root_properties :
  (Real.sqrt 285.61 = 16.9) ∧
  (Int.sqrt 26896 = 164) ∧
  (∃! (s : Finset ℕ), (∀ n ∈ s, 16.1 < Real.sqrt n ∧ Real.sqrt n < 16.2) ∧ s.card = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_properties_l993_99302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l993_99331

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a/(2^x)

noncomputable def C1 (a : ℝ) (x : ℝ) : ℝ := f a (x - 2)

noncomputable def C2 (a : ℝ) (x : ℝ) : ℝ := -C1 a x + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := C2 a x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x / a + g a x

def has_minimum (F : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, F x ≥ m

theorem a_range (a : ℝ) :
  (∃ m : ℝ, has_minimum (F a) ∧ m > 2 + Real.sqrt 7) →
  1/2 < a ∧ a < 2 := by
  sorry

#check a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l993_99331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_l993_99329

/-- The parabola is defined by the equation y = 2x^2 -/
def parabola (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y = 2 * x^2

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (f : ℝ × ℝ) (p : ℝ × ℝ → Prop) : Prop :=
  p f ∧ ∀ (x' : ℝ × ℝ), p x' → f = x'

/-- Theorem: The coordinates of the focus of the parabola y = 2x^2 are (0, 1/8) -/
theorem focus_of_parabola :
  is_focus (0, 1/8) parabola := by
  sorry

#check focus_of_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_l993_99329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_zero_of_polynomial_l993_99300

def is_valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + (P 0)

theorem complex_zero_of_polynomial 
  (P : ℝ → ℝ) 
  (h_valid : is_valid_polynomial P) 
  (h_zeros : ∃ (z₁ z₂ : ℤ), P z₁ = 0 ∧ P z₂ = 0 ∧ z₁ ≠ z₂) :
  ∃ (z : ℂ), z = Complex.mk (-3/2) (Real.sqrt 15 / 2) ∧ 
    (Complex.mk (P z.re) 0).im = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_zero_of_polynomial_l993_99300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_symmetry_l993_99399

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem cosine_function_symmetry 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : abs φ < Real.pi / 2) 
  (h_axis : ∀ x, f ω φ (x + Real.pi / 3) = f ω φ (Real.pi / 3 - x)) 
  (h_dist : Real.pi / (4 * ω) = Real.pi / 8) :
  ∀ x, f ω φ x = Real.cos (4 * x + Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_symmetry_l993_99399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compounding_difference_for_maria_l993_99397

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Calculates the difference between semi-annual and annual compounding -/
noncomputable def compounding_difference (principal : ℝ) (annual_rate : ℝ) (years : ℕ) : ℝ :=
  let semi_annual := compound_interest principal (annual_rate / 2) (2 * years)
  let annual := compound_interest principal annual_rate years
  semi_annual - annual

theorem compounding_difference_for_maria : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |compounding_difference 8000 0.1 3 - 72.80| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compounding_difference_for_maria_l993_99397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_completion_time_l993_99356

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  initialWorkers : ℕ
  completedLength : ℝ
  daysWorked : ℕ
  extraWorkers : ℕ

/-- Calculates the initial planned completion time for a road project -/
noncomputable def initialPlannedDays (project : RoadProject) : ℝ :=
  (project.totalLength * project.daysWorked * (project.initialWorkers + project.extraWorkers : ℝ)) /
  (project.completedLength * (project.initialWorkers + project.extraWorkers : ℝ))

theorem road_project_completion_time :
  let project : RoadProject := {
    totalLength := 10,
    initialWorkers := 30,
    completedLength := 2,
    daysWorked := 50,
    extraWorkers := 30
  }
  initialPlannedDays project = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_completion_time_l993_99356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l993_99323

/-- The time (in hours) that P and Q worked together on the job -/
noncomputable def time_worked_together : ℝ := 2

/-- P's work rate in jobs per hour -/
noncomputable def p_rate : ℝ := 1/3

/-- Q's work rate in jobs per hour -/
noncomputable def q_rate : ℝ := 1/18

/-- The combined work rate of P and Q when working together -/
noncomputable def combined_rate : ℝ := p_rate + q_rate

/-- The time (in hours) it takes P to finish the remaining part of the job alone -/
noncomputable def p_remaining_time : ℝ := 40/60

theorem job_completion_time : 
  time_worked_together * combined_rate + p_remaining_time * p_rate = 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l993_99323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_even_count_l993_99352

theorem four_digit_even_count : 
  (Finset.filter (fun n => n ≥ 1000 ∧ n < 2000 ∧ Even n) (Finset.range 10000)).card = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_even_count_l993_99352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_area_between_parallel_lines_l993_99379

/-- The distance between two parallel lines with equations ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

/-- The area of a circle with radius r -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem max_circle_area_between_parallel_lines :
  let l₁ : ℝ → ℝ → ℝ := λ x y ↦ 3*x - 4*y
  let l₂ : ℝ → ℝ → ℝ := λ x y ↦ 3*x - 4*y - 20
  let d := distance_between_parallel_lines 3 (-4) 0 (-20)
  let max_area := circle_area (d/2)
  max_area = 4*Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_area_between_parallel_lines_l993_99379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_throws_l993_99314

/-- The radius of the dartboard -/
noncomputable def dartboard_radius : ℝ := 20

/-- The radius of the target area -/
noncomputable def target_radius : ℝ := 10

/-- The number of darts initially on the board -/
def initial_darts : ℕ := 2020

/-- The probability that a dart lands within the target area -/
noncomputable def target_probability : ℝ := (target_radius / dartboard_radius) ^ 2

/-- The expected number of throws for a single dart to be within the target area -/
noncomputable def expected_throws_per_dart : ℝ := 1 / target_probability

/-- The theorem stating the expected number of throws to have all darts within the target area -/
theorem expected_total_throws : 
  ∀ (r R : ℝ) (n : ℕ),
  r = target_radius → 
  R = dartboard_radius → 
  n = initial_darts →
  (r / R) ^ 2 = target_probability →
  1 / ((r / R) ^ 2) = expected_throws_per_dart →
  n * expected_throws_per_dart = 6060 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_throws_l993_99314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_a_bound_f_max_value_a_range_l993_99310

/-- The function f(x) = x^2 / (x + 3/2) -/
noncomputable def f (x : ℝ) : ℝ := x^2 / (x + 3/2)

/-- The theorem stating that if x^2 - 2ax - 3a ≤ 0 holds for all x in [-1, 2], then a ≥ 1 -/
theorem inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, x^2 - 2*a*x - 3*a ≤ 0) →
  a ≥ 1 := by
  sorry

/-- The maximum value of f(x) over the interval [-1, 2] is 2 -/
theorem f_max_value :
  ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ 2 := by
  sorry

/-- The range of possible values for a is [1, +∞) -/
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, x^2 - 2*a*x - 3*a ≤ 0) ↔
  a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_a_bound_f_max_value_a_range_l993_99310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_profit_is_eight_percent_l993_99370

/-- Calculates the profit percentage for tomato sales given the initial purchase price,
    percentage of ruined tomatoes, and final selling price. -/
noncomputable def profit_percentage (initial_price : ℝ) (ruined_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let remaining_percentage := 1 - ruined_percentage
  let revenue := selling_price * remaining_percentage
  let profit := revenue - initial_price
  (profit / initial_price) * 100

/-- Theorem stating that the profit percentage for the given scenario is 8% -/
theorem tomato_profit_is_eight_percent :
  profit_percentage 0.80 0.20 1.08 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_profit_is_eight_percent_l993_99370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l993_99344

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/6) + Real.cos x^4 - Real.sin x^4

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (-Real.pi/12) (Real.pi/6), f x ≥ (Real.sqrt 3 + 1) / 2) ∧
  (f (-Real.pi/12) = (Real.sqrt 3 + 1) / 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi/12) (Real.pi/6), f x ≤ Real.sqrt 3 + 1/2) ∧
  (f (Real.pi/6) = Real.sqrt 3 + 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l993_99344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_inequality_l993_99360

/-- Predicate to represent that A, B, C, D are valid face areas of a tetrahedron -/
def IsTetrahedronFaceAreas (A B C D : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
  A + B > C + D ∧ A + C > B + D ∧ A + D > B + C ∧
  B + C > A + D ∧ B + D > A + C ∧ C + D > A + B

theorem tetrahedron_face_area_inequality 
  (A B C D : ℝ) 
  (hA : A ≥ 0) (hB : B ≥ 0) (hC : C ≥ 0) (hD : D ≥ 0) 
  (h_tetrahedron : IsTetrahedronFaceAreas A B C D) : 
  |A^2 + B^2 - C^2 - D^2| ≤ 2*(A*B + C*D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_inequality_l993_99360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_largest_area_l993_99355

-- Define the shapes
def triangle_ABC (A B C : ℝ) : Prop := 
  A = 60 ∧ B = 45 ∧ C = Real.sqrt 2

def trapezoid (d1 d2 angle : ℝ) : Prop := 
  d1 = Real.sqrt 2 ∧ d2 = Real.sqrt 3 ∧ angle = 75

def circle_shape (r : ℝ) : Prop := r = 1

def square (d : ℝ) : Prop := d = 2.5

-- Define area calculation functions
noncomputable def area_circle (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def area_triangle (A B C : ℝ) : ℝ := sorry
noncomputable def area_trapezoid (d1 d2 angle : ℝ) : ℝ := sorry
noncomputable def area_square (d : ℝ) : ℝ := (d / Real.sqrt 2)^2

-- State the theorem
theorem circle_largest_area 
  (t : ℝ → ℝ → ℝ → Prop) 
  (trap : ℝ → ℝ → ℝ → Prop) 
  (c : ℝ → Prop) 
  (s : ℝ → Prop) : 
  (∀ A B C, t A B C → triangle_ABC A B C) →
  (∀ d1 d2 angle, trap d1 d2 angle → trapezoid d1 d2 angle) →
  (∀ r, c r → circle_shape r) →
  (∀ d, s d → square d) →
  (∀ A B C d1 d2 angle r d, 
    t A B C → 
    trap d1 d2 angle → 
    c r → 
    s d → 
    area_circle r > area_triangle A B C ∧
    area_circle r > area_trapezoid d1 d2 angle ∧
    area_circle r > area_square d) :=
by
  sorry

-- Note: The actual implementations of area_triangle and area_trapezoid
-- are left as 'sorry' as they would require more complex calculations.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_largest_area_l993_99355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l993_99380

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement
theorem f_properties :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧ 
  f 1 = -1 ∧
  (∀ x y, x < y → x < -1/3 → f x < f y) ∧
  (∀ x y, x < y → 1 < x → f x < f y) ∧
  (∀ x y, -1/3 < x → x < y → y < 1 → f x > f y) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l993_99380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excellence_frequency_calculation_l993_99321

/-- Represents the frequency distribution of long jump distances --/
structure LongJumpData :=
  (freq : Fin 5 → ℕ)
  (excellent_threshold : ℝ)

/-- Calculates the frequency of excellence in long jump --/
def excellence_frequency (data : LongJumpData) : ℚ :=
  let excellent_freq := data.freq 3 + data.freq 4
  let total_freq := (Finset.range 5).sum (λ i => ↑(data.freq i))
  ↑excellent_freq / ↑total_freq

/-- The given long jump data for the class --/
def class_data : LongJumpData :=
  { freq := λ i => [1, 4, 8, 10, 2].get i,
    excellent_threshold := 1.8 }

theorem excellence_frequency_calculation :
  excellence_frequency class_data = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excellence_frequency_calculation_l993_99321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_iff_a_eq_half_l993_99303

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.sin x - a * x^2 - (1 + a) * x

-- State the theorem
theorem f_monotone_increasing_iff_a_eq_half :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_iff_a_eq_half_l993_99303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l993_99363

theorem evaluate_expression : (64 : ℝ) ^ (-(2 : ℝ) ^ (-(3 : ℝ))) = 1 / (8 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l993_99363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l993_99372

/-- Represents the composition of an acid-water mixture -/
structure Mixture where
  acid : ℚ
  water : ℚ

/-- The original mixture -/
noncomputable def original : Mixture :=
  { acid := 1, water := 3 }

/-- The mixture after adding 1 oz of water -/
noncomputable def after_water : Mixture :=
  { acid := original.acid,
    water := original.water + 1 }

/-- The mixture after adding 1 oz of acid to the previous mixture -/
noncomputable def after_acid : Mixture :=
  { acid := after_water.acid + 1,
    water := after_water.water }

/-- The percentage of acid in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℚ :=
  m.acid / (m.acid + m.water) * 100

theorem original_mixture_acid_percentage :
  acid_percentage after_water = 20 ∧
  acid_percentage after_acid = 100/3 →
  acid_percentage original = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l993_99372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_correct_l993_99309

def initial_spoons : ℕ := 26
def final_spoons : ℕ := 33696

def is_valid_sequence (seq : List ℕ) : Prop :=
  seq.head? = some initial_spoons ∧
  seq.getLast? = some final_spoons ∧
  ∀ i, i < seq.length - 1 → (seq.get! (i+1) = 2 * seq.get! i ∨ seq.get! (i+1) = 3 * seq.get! i)

def min_steps : ℕ := 9

theorem min_steps_correct : 
  (∀ seq : List ℕ, is_valid_sequence seq → seq.length ≥ min_steps + 1) ∧
  ∃ seq : List ℕ, is_valid_sequence seq ∧ seq.length = min_steps + 1 :=
by sorry

#check min_steps_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_correct_l993_99309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_a_finish_time_l993_99339

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runnerA : Runner
  runnerB : Runner
  finishTimeB : ℝ
  leadDistance : ℝ

/-- Calculates the finish time for runner A given the race parameters -/
noncomputable def calculateFinishTimeA (race : Race) : ℝ :=
  (race.distance - race.leadDistance) / (race.distance / race.finishTimeB)

/-- Theorem stating that under the given conditions, runner A finishes in 36 seconds -/
theorem runner_a_finish_time (race : Race) 
  (h1 : race.distance = 100) 
  (h2 : race.finishTimeB = 45) 
  (h3 : race.leadDistance = 20) : 
  calculateFinishTimeA race = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_a_finish_time_l993_99339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l993_99332

noncomputable def f (x : ℝ) := 2 * Real.sin (3 * x + Real.pi / 3)

theorem f_extrema :
  (∀ x, f x ≤ 2) ∧
  (∀ x, f x ≥ -2) ∧
  (∀ k : ℤ, f ((2 * k * Real.pi / 3) + Real.pi / 18) = 2) ∧
  (∀ k : ℤ, f ((2 * k * Real.pi / 3) - 5 * Real.pi / 18) = -2) ∧
  (∀ x, f x = 2 → ∃ k : ℤ, x = (2 * k * Real.pi / 3) + Real.pi / 18) ∧
  (∀ x, f x = -2 → ∃ k : ℤ, x = (2 * k * Real.pi / 3) - 5 * Real.pi / 18) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l993_99332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l993_99317

noncomputable def f (a b x : ℝ) := a * x^2 + b * x

noncomputable def F (a x : ℝ) := 4 * f (-1/2) (1/2) (a^x) + 3 * a^(2*x) - 1

theorem quadratic_function_properties 
  (a b : ℝ) 
  (h1 : ∀ x, f a b (x-1) = f a b x + x - 1) :
  (∀ x, f a b x = -1/2 * x^2 + 1/2 * x) ∧ 
  (∀ x, f a b x = 0 ↔ x = 0 ∨ x = 1) ∧
  (∀ x, f a b x < 0 ↔ x > 1 ∨ x < 0) ∧
  (∀ a, a > 0 → a ≠ 1 → 
    (∀ x, x ∈ Set.Icc (-1) 1 → F a x ≤ 14) →
    (∃ x, x ∈ Set.Icc (-1) 1 ∧ F a x = 14) →
    (a = 1/3 ∨ a = 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l993_99317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_point_l993_99348

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Power of a point with respect to a circle -/
theorem power_of_point (c : Circle) (M A B : Point) :
  M ∉ {p : Point | distance p c.center = c.radius} →
  distance M c.center > c.radius →
  (∃ (l : Set Point), A ∈ l ∧ B ∈ l ∧ M ∈ l ∧
    A ∈ {p : Point | distance p c.center = c.radius} ∧
    B ∈ {p : Point | distance p c.center = c.radius}) →
  distance M A * distance M B = (distance M c.center)^2 - c.radius^2 := by
  sorry

#check power_of_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_point_l993_99348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberry_calculation_l993_99342

/-- Calculates the amount of frozen blueberries given total production and percentages --/
def frozen_blueberries (total : ℝ) (mixed_percent : ℝ) (frozen_percent : ℝ) : ℝ :=
  let remaining := total * (1 - mixed_percent)
  remaining * frozen_percent

/-- Rounds a number to the nearest tenth --/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem blueberry_calculation :
  let total := 4.8
  let mixed_percent := 0.25
  let frozen_percent := 0.40
  round_to_tenth (frozen_blueberries total mixed_percent frozen_percent) = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberry_calculation_l993_99342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l993_99325

/-- Sequence b defined recursively -/
def b : ℕ → ℚ
  | 0 => 2  -- Changed 1 to 0 for zero-based indexing
  | 1 => 3  -- Changed 2 to 1
  | n + 2 => (1 / 2) * b (n + 1) + (1 / 5) * b n

/-- Sum of the sequence b -/
noncomputable def seriesSum : ℚ := ∑' n, b n

/-- Theorem: The sum of the sequence b₀ + b₁ + b₂ + ... equals 40/3 -/
theorem sum_of_sequence_b : seriesSum = 40 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l993_99325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_15_terms_is_60_l993_99334

/-- An arithmetic progression with the sum of the 4th and 12th terms equal to 8 -/
def ArithmeticProgression (a : ℝ) (d : ℝ) : Prop :=
  (a + 3*d) + (a + 11*d) = 8

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def SumOfTerms (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2*a + (n - 1 : ℝ)*d)

/-- Theorem: The sum of the first 15 terms is 60 -/
theorem sum_of_first_15_terms_is_60 (a d : ℝ) (h : ArithmeticProgression a d) :
  SumOfTerms a d 15 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_15_terms_is_60_l993_99334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_harmonic_numbers_eq_255024_l993_99330

/-- A harmonic number is a positive integer that can be expressed as the difference
    of squares of two consecutive odd numbers. -/
def IsHarmonicNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 1)^2 - (2*k - 1)^2

/-- The sum of all harmonic numbers not exceeding 2023 -/
def SumOfHarmonicNumbers : ℕ :=
  (Finset.range 254).sum (fun i => 8 * (i + 1))

theorem sum_of_harmonic_numbers_eq_255024 : SumOfHarmonicNumbers = 255024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_harmonic_numbers_eq_255024_l993_99330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_equation_for_long_trips_l993_99376

/-- Represents the fare calculation for a taxi trip -/
noncomputable def taxi_fare (x : ℝ) : ℝ :=
  if x ≤ 3 then 8 else 8 + 2.7 * (x - 3)

/-- Theorem stating the relationship between fare and distance for trips over 3 km -/
theorem fare_equation_for_long_trips (x : ℝ) (h : x > 3) :
  taxi_fare x = 2.7 * x - 0.1 := by
  sorry

#check fare_equation_for_long_trips

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_equation_for_long_trips_l993_99376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_region_implies_a_eq_neg_four_l993_99364

noncomputable def f (a b c x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x + c)

def D (a b c : ℝ) : Set ℝ := {x | a * x^2 + b * x + c ≥ 0}

theorem square_region_implies_a_eq_neg_four (a b c : ℝ) (ha : a < 0) :
  (∀ s t, s ∈ D a b c → t ∈ D a b c → 
    ∃ l, Set.Icc 0 l ×ˢ Set.Icc 0 l = {p : ℝ × ℝ | p.1 ∈ D a b c ∧ p.2 = f a b c p.1}) →
  a = -4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_region_implies_a_eq_neg_four_l993_99364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l993_99393

noncomputable def f (x : ℝ) : ℝ := Real.sin (2*x + Real.pi/3) * Real.cos (x - Real.pi/6) + Real.cos (2*x + Real.pi/3) * Real.sin (x - Real.pi/6)

theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (Real.pi - x) = f (Real.pi + x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l993_99393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l993_99371

theorem quadratic_root_difference (k : ℝ) : 
  (∃ c₁ c₂ : ℝ, c₁ > c₂ ∧ 
   c₁ = (-5 + Real.sqrt (25 + 8*k)) / 4 ∧ 
   c₂ = (-5 - Real.sqrt (25 + 8*k)) / 4 ∧
   2 * c₁^2 + 5 * c₁ = k ∧
   2 * c₂^2 + 5 * c₂ = k ∧
   c₁ - c₂ = 5.5) → 
  k = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l993_99371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_12_in_20_factorial_l993_99345

theorem highest_power_of_12_in_20_factorial : 
  ∃ k : ℕ, k = 8 ∧ 12^k ∣ Nat.factorial 20 ∧ ∀ m : ℕ, 12^m ∣ Nat.factorial 20 → m ≤ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_12_in_20_factorial_l993_99345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_log2_l993_99306

-- Define the function f(x) = log₂(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem min_value_log2 :
  ∀ x > 0, f x ≥ 0 ∧ f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_log2_l993_99306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expo_volunteer_assignment_l993_99305

/-- Represents the number of volunteers --/
def total_volunteers : ℕ := 5

/-- Represents the number of tasks --/
def number_of_tasks : ℕ := 3

/-- Represents the number of volunteers to be selected --/
def selected_volunteers : ℕ := 3

/-- Represents whether a specific volunteer can't do a specific task --/
def has_restriction : Bool := true

/-- Calculates the number of assignment schemes --/
def number_of_assignment_schemes (total : ℕ) (tasks : ℕ) (selected : ℕ) (restriction : Bool) : ℕ :=
  if restriction then
    (Nat.choose (total - 1) selected * Nat.factorial selected) +
    (Nat.choose (total - 1) (selected - 1) * Nat.factorial (selected - 1))
  else
    Nat.choose total selected * Nat.factorial selected

theorem expo_volunteer_assignment :
  (total_volunteers = 5) →
  (number_of_tasks = 3) →
  (selected_volunteers = 3) →
  has_restriction →
  (∃ (n : ℕ), n = 48 ∧ 
    n = number_of_assignment_schemes total_volunteers number_of_tasks selected_volunteers has_restriction) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expo_volunteer_assignment_l993_99305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_tennis_percentage_approx_52_l993_99336

/-- Represents the total number of students at North High School -/
def north_students : ℕ := 1800

/-- Represents the total number of students at South Academy -/
def south_students : ℕ := 3000

/-- Represents the percentage of students who prefer tennis at North High School -/
def north_tennis_percentage : ℚ := 25 / 100

/-- Represents the percentage of students who prefer tennis at South Academy -/
def south_tennis_percentage : ℚ := 35 / 100

/-- Theorem stating that the combined percentage of students who prefer tennis at both schools is approximately 52% -/
theorem combined_tennis_percentage_approx_52 :
  let north_tennis := (north_students : ℚ) * north_tennis_percentage
  let south_tennis := (south_students : ℚ) * south_tennis_percentage
  let total_tennis := north_tennis + south_tennis
  let total_students := (north_students + south_students : ℚ)
  let combined_percentage := total_tennis / total_students * 100
  ∃ ε > 0, |combined_percentage - 52| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_tennis_percentage_approx_52_l993_99336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_irrational_square_rational_l993_99347

theorem negation_irrational_square_rational :
  ¬(∃ x : ℝ, Irrational x ∧ (∃ q : ℚ, x^2 = q)) ↔
  ∀ x : ℝ, Irrational x → Irrational (x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_irrational_square_rational_l993_99347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_A_intersect_C_empty_iff_l993_99353

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y | ∃ x > 0, y = Real.log (x + 1)}

-- Define set B
def B : Set ℝ := {x | (1/2 : ℝ) ≤ (2 : ℝ) ^ x ∧ (2 : ℝ) ^ x ≤ 8}

-- Define set C (parameterized by a)
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a}

-- Statement 1: (∁ᵤA) ∪ B = (-∞, 3]
theorem complement_A_union_B : (Set.univ \ A) ∪ B = Set.Iic 3 := by sorry

-- Statement 2: A ∩ C = ∅ if and only if a ∈ (-∞, 0]
theorem A_intersect_C_empty_iff (a : ℝ) : A ∩ C a = ∅ ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_A_intersect_C_empty_iff_l993_99353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_point_optimal_decision_profit_comparison_l993_99375

/-- Represents the decision to sell a commodity -/
inductive SellDecision
| Beginning
| End

/-- Determines the optimal selling decision based on the commodity cost -/
noncomputable def optimalSellDecision (cost : ℝ) : SellDecision :=
  if cost < 525 then SellDecision.Beginning else SellDecision.End

theorem break_even_point (cost : ℝ) :
  let profit_beginning := 100 + (cost + 100) * 0.024
  let profit_end := 120 - 5
  profit_beginning = profit_end ↔ cost = 525 := by sorry

theorem optimal_decision (cost : ℝ) :
  optimalSellDecision cost = SellDecision.Beginning ↔ cost < 525 := by sorry

theorem profit_comparison (cost : ℝ) :
  let profit_beginning := 100 + (cost + 100) * 0.024
  let profit_end := 120 - 5
  (profit_beginning > profit_end ↔ cost < 525) ∧
  (profit_beginning < profit_end ↔ cost > 525) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_point_optimal_decision_profit_comparison_l993_99375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_power_minus_one_l993_99315

theorem divides_power_minus_one (k n : ℕ) :
  (3^k : ℕ) ∣ (2^n : ℕ) - 1 ↔ (2 * 3^(k-1) : ℕ) ∣ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_power_minus_one_l993_99315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l993_99394

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y + 1 = 0

-- Define the inclination angle
noncomputable def inclination_angle (eq : (ℝ → ℝ → Prop)) : ℝ := 
  Real.arctan (-1 / Real.sqrt 3)

-- Theorem statement
theorem line_inclination_angle :
  inclination_angle line_equation = π * 5 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l993_99394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l993_99324

/-- 
Given a triangle ABC where:
- a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively
- a^2 + b^2 < c^2
- sin C = √3/2
Then, angle C = 2π/3
-/
theorem angle_C_measure (a b c : ℝ) (h1 : a^2 + b^2 < c^2) (h2 : Real.sin C = Real.sqrt 3 / 2) :
  C = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l993_99324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_is_sin_l993_99346

-- Define the recursive function
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => Real.cos
  | n + 1 => deriv (f n)

-- State the theorem
theorem f_2011_is_sin : f 2011 = Real.sin := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_is_sin_l993_99346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_z_fourth_eq_neg_sixteen_l993_99357

/-- The set of complex numbers z satisfying z^4 = -16 -/
def SolutionSet : Set ℂ :=
  {z : ℂ | z^4 = -16}

/-- The set of four specific complex numbers -/
noncomputable def ExpectedSet : Set ℂ :=
  {Complex.mk (Real.sqrt 2) (Real.sqrt 2), 
   Complex.mk (Real.sqrt 2) (-Real.sqrt 2), 
   Complex.mk (-Real.sqrt 2) (Real.sqrt 2), 
   Complex.mk (-Real.sqrt 2) (-Real.sqrt 2)}

theorem solutions_to_z_fourth_eq_neg_sixteen :
  SolutionSet = ExpectedSet := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_z_fourth_eq_neg_sixteen_l993_99357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_borrowed_amount_l993_99301

-- Define the interest rate
def interest_rate : ℚ := 4 / 100

-- Define the loan duration in years
def loan_duration : ℕ := 2

-- Define the final amount
def final_amount : ℚ := 8112

-- Define the compound interest formula
noncomputable def compound_interest (principal : ℚ) : ℚ :=
  principal * (1 + interest_rate) ^ loan_duration

-- Theorem statement
theorem initial_borrowed_amount :
  ∃ (principal : ℚ), compound_interest principal = final_amount ∧ principal = 7500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_borrowed_amount_l993_99301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_simplest_abs_not_simplest_l993_99327

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ y : ℝ, x = Real.sqrt (y^2)

-- Define the concept of simplicity for quadratic radicals
def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧
  ∀ y : ℝ, QuadraticRadical y → (∃ z : ℝ, z ≠ 1 ∧ y = z * x) → x < y

-- The theorem statement
theorem sqrt_2_simplest :
  IsSimplestQuadraticRadical (Real.sqrt 2) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt (1/5)) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt 27) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt 1) := by
  sorry

-- Helper theorem for the last part of the original statement
theorem abs_not_simplest (a : ℝ) :
  ¬IsSimplestQuadraticRadical (|a|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_simplest_abs_not_simplest_l993_99327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l993_99359

theorem angle_in_fourth_quadrant (θ : ℝ) 
  (h1 : Real.cos θ > 0) 
  (h2 : Real.sin (2 * θ) < 0) : 
  θ % (2 * Real.pi) ∈ Set.Ioo ((3 * Real.pi) / 2) (2 * Real.pi) := by
  sorry

#check angle_in_fourth_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l993_99359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_player_counts_l993_99304

/-- Represents a group of players in a game -/
structure PlayerGroup where
  /-- Total number of players -/
  total : ℕ
  /-- Number of groups the players are divided into -/
  num_groups : ℕ
  /-- Number of players in each group -/
  players_per_group : ℕ
  /-- Condition: Total players is the product of number of groups and players per group -/
  total_eq : total = num_groups * players_per_group
  /-- Condition: Each player has exactly 15 opponents -/
  opponents_eq : (num_groups - 1) * players_per_group = 15

/-- Theorem stating the possible total number of players -/
theorem valid_player_counts : 
  ∀ g : PlayerGroup, g.total ∈ ({16, 18, 20, 30} : Set ℕ) := by
  sorry

#check valid_player_counts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_player_counts_l993_99304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OA_perp_OB_circle_through_origin_circle_equation_l993_99351

-- Define the points M and N
def M : ℝ × ℝ := (1, -3)
def N : ℝ × ℝ := (5, 1)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point C
def C (t : ℝ) : ℝ × ℝ := (t * M.1 + (1-t) * N.1, t * M.2 + (1-t) * N.2)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Assume A and B are on the parabola and on the line through M and N
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2
axiom A_on_line : ∃ t : ℝ, A = C t
axiom B_on_line : ∃ t : ℝ, B = C t

-- Theorem 1: OA is perpendicular to OB
theorem OA_perp_OB : A.1 * B.1 + A.2 * B.2 = 0 := by sorry

-- Define point P
def P : ℝ × ℝ := (4, 0)

-- Define points D and E
noncomputable def D : ℝ × ℝ := sorry
noncomputable def E : ℝ × ℝ := sorry

-- Assume D and E are on the parabola and on a line through P
axiom D_on_parabola : parabola D.1 D.2
axiom E_on_parabola : parabola E.1 E.2
axiom DE_through_P : ∃ k : ℝ, D.1 = k * D.2 + 4 ∧ E.1 = k * E.2 + 4

-- Theorem 2: Circle with diameter DE passes through the origin
theorem circle_through_origin : D.1 * E.1 + D.2 * E.2 = 0 := by sorry

-- Define the circle's trajectory
def circle_trajectory (x y : ℝ) : Prop := y^2 = 2*x - 8

-- Theorem 3: The equation of the circle's trajectory
theorem circle_equation : ∀ x y : ℝ, 
  (∃ t : ℝ, (x, y) = ((D.1 + E.1)/2, (D.2 + E.2)/2)) → circle_trajectory x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_OA_perp_OB_circle_through_origin_circle_equation_l993_99351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l993_99398

/-- The total distance traveled in a trip with three equal parts, each taking 15 minutes,
    with speeds of 16 mph, 12 mph, and 20 mph respectively, is 12 miles. -/
theorem trip_distance (time_per_part : ℝ) (speed1 : ℝ) (speed2 : ℝ) (speed3 : ℝ)
  (h1 : time_per_part = 15 / 60)
  (h2 : speed1 = 16)
  (h3 : speed2 = 12)
  (h4 : speed3 = 20) :
  speed1 * time_per_part + speed2 * time_per_part + speed3 * time_per_part = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l993_99398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_7x14_grid_l993_99307

/-- Represents a rectangular grid -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a partition of the grid into squares and L-shaped pieces -/
structure Partition where
  squares : ℕ
  l_pieces : ℕ

/-- Checks if a partition is valid for a given grid -/
def is_valid_partition (g : Grid) (p : Partition) : Prop :=
  4 * p.squares + 3 * p.l_pieces = g.rows * g.cols

/-- The main theorem about partitioning a 7x14 grid -/
theorem partition_7x14_grid :
  let g : Grid := ⟨7, 14⟩
  (∃ p : Partition, is_valid_partition g p ∧ p.squares = p.l_pieces) ∧
  (¬ ∃ p : Partition, is_valid_partition g p ∧ p.squares > p.l_pieces) := by
  sorry

#check partition_7x14_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_7x14_grid_l993_99307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_l993_99388

/-- Triangle ABC with vertices A(0, 10), B(4, 0), C(10, 0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 10⟩, ⟨4, 0⟩, ⟨10, 0⟩}

/-- Point T on line AB -/
noncomputable def point_T (t : ℝ) : ℝ × ℝ :=
  ⟨4 - 2*t/5, t⟩

/-- Point U on line AC -/
noncomputable def point_U (t : ℝ) : ℝ × ℝ :=
  ⟨10 - t, t⟩

/-- Area of triangle ATU -/
noncomputable def area_ATU (t : ℝ) : ℝ :=
  (1/2) * ((3*t/5 + 6) * (10 - t))

/-- The main theorem -/
theorem horizontal_line_intersection (t : ℝ) :
  t ∈ Set.Ioo 0 10 →
  area_ATU t = 15 →
  t = 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_l993_99388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_theorem_l993_99349

/-- Represents the inverse proportion function y = 2/x -/
noncomputable def inverse_proportion (x : ℝ) : ℝ := 2 / x

theorem inverse_proportion_range_theorem (m n : ℝ) :
  (n = inverse_proportion m) → (n ≥ -1) → (m ≤ -2 ∨ m > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_theorem_l993_99349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_construction_l993_99384

/-- The side length of the smallest square that can be constructed from an equal number of squares with sides 1, 2, and 3. -/
def smallest_square_side : ℕ := 14

/-- The number of squares of each size needed to construct the smallest possible square. -/
def num_squares : ℕ := 14

theorem smallest_square_construction :
  (num_squares * (1^2 + 2^2 + 3^2) : ℕ) = smallest_square_side^2 ∧
  ∀ n : ℕ, n < num_squares → ∀ m : ℕ, n * (1^2 + 2^2 + 3^2) ≠ m^2 :=
by sorry

#check smallest_square_construction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_construction_l993_99384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l993_99381

-- Define the triangle ABC
variable (A B C H : ℝ × ℝ)

-- Define the vectors
def AB (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def BC (B C : ℝ × ℝ) : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def CA (C A : ℝ × ℝ) : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
def AH (A H : ℝ × ℝ) : ℝ × ℝ := (H.1 - A.1, H.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the conditions
def condition1 (C : ℝ × ℝ) : Prop := Real.cos (C.1 / 2) = 2 * Real.sqrt 5 / 5
def condition2 (A H B C : ℝ × ℝ) : Prop := dot_product (AH A H) (BC B C) = 0
def condition3 (A B C : ℝ × ℝ) : Prop := dot_product (AB A B) (CA C A + BC B C) = 0

-- Define distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define eccentricity
noncomputable def eccentricity (A H C : ℝ × ℝ) : ℝ := distance A H / (distance A C - distance C H)

-- Theorem statement
theorem hyperbola_eccentricity (A B C H : ℝ × ℝ) :
  condition1 C → condition2 A H B C → condition3 A B C →
  eccentricity A H C = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l993_99381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l993_99390

/-- The parabola equation y = x^2 - 7x + 12 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 7*x + 12

/-- The area of a triangle given three points -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1))

theorem max_triangle_area :
  ∀ p r : ℝ,
  2 ≤ p → p ≤ 5 →
  parabola 2 5 →
  parabola 5 4 →
  parabola p r →
  (∀ q s : ℝ, 2 ≤ q → q ≤ 5 → parabola q s →
    triangleArea 2 5 5 4 q s ≤ triangleArea 2 5 5 4 p r) →
  triangleArea 2 5 5 4 p r = 1281/72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l993_99390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_period_4pi_l993_99382

/-- The period of a cosine function with frequency k and phase shift π/6 -/
noncomputable def cosPeriod (k : ℝ) : ℝ := 2 * Real.pi / k

/-- Theorem: If the period of cos(kx + π/6) is 4π, then k = 1/2 -/
theorem cos_period_4pi (k : ℝ) (h1 : k > 0) (h2 : cosPeriod k = 4 * Real.pi) : k = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_period_4pi_l993_99382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l993_99362

/-- An odd function f(x) with specific properties -/
noncomputable def f (x : ℝ) : ℝ := -x + 2/x

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  f 1 = 1 ∧
  f 2 = -1 ∧
  (∀ x > 0, ∀ y > x, f y < f x) ∧  -- f is decreasing on (0, +∞)
  (∀ t, (∀ x ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2, |t - 1| ≤ f x + 2) → t ∈ Set.Icc 0 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l993_99362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_fuel_theorem_l993_99383

/-- Represents the properties of a truck's fuel consumption and diesel cost -/
structure TruckFuel where
  miles : ℚ
  gallons : ℚ
  cost_per_gallon : ℚ

/-- Calculates the distance a truck can travel given a certain amount of fuel -/
def distance_for_fuel (tf : TruckFuel) (new_gallons : ℚ) : ℚ :=
  (tf.miles / tf.gallons) * new_gallons

/-- Calculates the cost for a given amount of fuel -/
def cost_for_fuel (tf : TruckFuel) (new_gallons : ℚ) : ℚ :=
  tf.cost_per_gallon * new_gallons

/-- Theorem stating the distance and cost for a specific truck and fuel amount -/
theorem truck_fuel_theorem (tf : TruckFuel) 
    (h1 : tf.miles = 240)
    (h2 : tf.gallons = 5)
    (h3 : tf.cost_per_gallon = 3)
    (new_gallons : ℚ)
    (h4 : new_gallons = 7) : 
    distance_for_fuel tf new_gallons = 336 ∧ 
    cost_for_fuel tf new_gallons = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_fuel_theorem_l993_99383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l993_99311

-- Define the function f based on the graph
noncomputable def f : ℝ → ℝ :=
  sorry  -- We don't provide an actual implementation, just a placeholder

-- Properties of f based on the graph
axiom f_piecewise (x : ℝ) :
  (x < -1 → f x = -2*x) ∧
  (-1 ≤ x ∧ x ≤ 3 → f x = x + 2) ∧
  (x > 3 → f x = -0.5*x + 4.5)

-- The main theorem to prove
theorem exactly_two_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l993_99311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_180_l993_99354

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors n)).sum id

theorem sum_of_odd_divisors_180 :
  sum_of_odd_divisors 180 = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_180_l993_99354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l993_99358

theorem expression_evaluation (b : ℝ) (hb : b ≠ 0) :
  (1 / 25) * (b ^ 0) + ((1 / (25 * b)) ^ 0) - (125 ^ (-1/3 : ℝ)) - ((-125) ^ (-1 : ℝ)) = 231 / 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l993_99358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_l993_99378

-- Define a type for points in a plane
variable (Point : Type)

-- Define a type for lines in a plane
variable (Line : Type)

-- Define a relation for a point being on a line
variable (on_line : Point → Line → Prop)

-- Define a relation for two lines being perpendicular
variable (perpendicular : Line → Line → Prop)

-- Theorem statement
theorem unique_perpendicular_line 
  (P : Point) (L : Line) (h : ¬ on_line P L) :
  ∃! M : Line, on_line P M ∧ perpendicular M L := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_l993_99378
