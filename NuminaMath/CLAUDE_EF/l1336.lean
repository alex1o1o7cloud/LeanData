import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_height_difference_l1336_133668

/-- Conversion factor from centimeters to inches -/
noncomputable def cm_to_inch : ℝ := 1 / 2.54

/-- Height of the old lamp in inches -/
def old_lamp_height : ℝ := 12

/-- Height of the new lamp in centimeters -/
def new_lamp_height_cm : ℝ := 55.56666666666667

/-- Height difference between the new and old lamps in inches -/
noncomputable def height_difference : ℝ := new_lamp_height_cm * cm_to_inch - old_lamp_height

theorem lamp_height_difference :
  ∀ ε > 0, |height_difference - 9.875| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_height_difference_l1336_133668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_catches_up_in_30_steps_l1336_133643

/-- Represents the ratio of steps between Xiaoming and his father -/
def xiaoming_father_step_ratio : ℚ := 8 / 5

/-- Represents the ratio of distance covered by father's steps to Xiaoming's steps -/
def father_xiaoming_distance_ratio : ℚ := 2 / 5

/-- Number of steps Xiaoming runs ahead before his father starts chasing -/
def xiaoming_head_start : ℕ := 27

/-- Calculates the number of steps father needs to run to catch up with Xiaoming -/
noncomputable def father_steps_to_catch_up : ℕ :=
  Int.toNat <| ⌈(xiaoming_head_start : ℚ) / (father_xiaoming_distance_ratio - xiaoming_father_step_ratio * father_xiaoming_distance_ratio)⌉

theorem father_catches_up_in_30_steps :
  father_steps_to_catch_up = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_catches_up_in_30_steps_l1336_133643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_intersect_T_eq_T_l1336_133670

-- Define set S
def S : Set Int := { s | ∃ n : Int, s = 2 * n + 1 }

-- Define set T
def T : Set Int := { t | ∃ n : Int, t = 4 * n + 1 }

-- Theorem statement
theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_intersect_T_eq_T_l1336_133670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_final_buttons_l1336_133615

def mark_buttons (initial : ℕ) (shane_multiplier : ℕ) (sam_fraction : ℚ) : ℕ :=
  let after_shane := initial + shane_multiplier * initial
  ((after_shane : ℚ) * (1 - sam_fraction)).floor.toNat

theorem mark_final_buttons :
  mark_buttons 14 3 (1/2) = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_final_buttons_l1336_133615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_length_l1336_133634

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (3, 0)

-- Define the line passing through the right focus with 30° inclination
noncomputable def line (x : ℝ) : ℝ := (x - 3) * Real.sqrt 3 / 3

-- Define points A and B as the intersection of the line and the hyperbola
noncomputable def point_A : ℝ × ℝ := (-3, -2 * Real.sqrt 3)
noncomputable def point_B : ℝ × ℝ := (9/5, -2 * Real.sqrt 3 / 5)

-- Theorem statement
theorem hyperbola_line_intersection_length :
  let A := point_A
  let B := point_B
  hyperbola A.1 A.2 ∧ 
  hyperbola B.1 B.2 ∧ 
  A.2 = line A.1 ∧ 
  B.2 = line B.1 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 / 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_length_l1336_133634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_neg_three_pi_four_l1336_133694

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x)

/-- The translated function g(x) -/
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6) + 1

/-- Theorem stating that g(-3π/4) = 3 -/
theorem g_at_neg_three_pi_four : g (-3 * Real.pi / 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_neg_three_pi_four_l1336_133694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_below_f_l1336_133656

/-- The function f(x) = x ln x + 3x - 2 -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 3 * x - 2

/-- The ray l(x) = kx - k -/
def l (k : ℝ) (x : ℝ) : ℝ := k * x - k

/-- Theorem stating that the maximum integer k such that l(k, x) < f(x) for all x ≥ 1 is 5 -/
theorem max_k_below_f : 
  (∃ k : ℤ, k = 5 ∧ 
    (∀ x : ℝ, x ≥ 1 → l (k : ℝ) x < f x) ∧ 
    (∀ m : ℤ, m > k → ∃ x : ℝ, x ≥ 1 ∧ l (m : ℝ) x ≥ f x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_below_f_l1336_133656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_is_approximately_77_92_l1336_133657

/-- Calculates the total amount made from selling baseball gear --/
noncomputable def total_amount_made (
  baseball_cards_price : ℝ)
  (baseball_cards_tax_rate : ℝ)
  (baseball_bat_price : ℝ)
  (baseball_glove_original_price : ℝ)
  (baseball_glove_discount_rate : ℝ)
  (baseball_glove_tax_rate : ℝ)
  (baseball_cleats_price : ℝ)
  (baseball_cleats_eur_tax_rate : ℝ)
  (usd_to_eur_rate : ℝ)
  (baseball_cleats_usd_discount_rate : ℝ)
  (baseball_cleats_usd_tax_rate : ℝ) : ℝ :=
  let baseball_cards := baseball_cards_price / (1 + baseball_cards_tax_rate)
  let baseball_bat := baseball_bat_price
  let baseball_glove := (baseball_glove_original_price * (1 - baseball_glove_discount_rate)) * (1 + baseball_glove_tax_rate)
  let baseball_cleats_eur := baseball_cleats_price / (1 + baseball_cleats_eur_tax_rate)
  let baseball_cleats_usd := (baseball_cleats_price * (1 - baseball_cleats_usd_discount_rate)) * (1 + baseball_cleats_usd_tax_rate)
  baseball_cards + baseball_bat + baseball_glove + baseball_cleats_eur + baseball_cleats_usd

/-- Theorem stating that the total amount made is approximately $77.92 --/
theorem total_amount_is_approximately_77_92 :
  ∃ ε > 0, |total_amount_made 25 0.05 10 30 0.20 0.08 10 0.10 0.85 0.15 0.07 - 77.92| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_is_approximately_77_92_l1336_133657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_identity_count_valid_n_l1336_133620

/-- The complex exponential identity for positive integers up to 1000 -/
theorem complex_exponential_identity (n : ℕ) (t : ℝ) (hn : 0 < n ∧ n ≤ 1000) :
  (Complex.exp (Complex.I * t)) ^ n = Complex.exp (Complex.I * (n : ℝ) * t) := by
  sorry

/-- Count of valid n values satisfying the complex exponential identity -/
theorem count_valid_n : 
  Finset.card (Finset.filter (fun n => 0 < n ∧ n ≤ 1000) (Finset.range 1001)) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_identity_count_valid_n_l1336_133620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1336_133696

/-- Given a line √2ax + by = 1 intersecting a circle x^2 + y^2 = 1, where triangle AOB is right-angled,
    prove that the minimum value of 1/a^2 + 2/b^2 is 4. -/
theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ (A B : ℝ × ℝ), 
    (Real.sqrt 2 * a * A.1 + b * A.2 = 1) ∧ 
    (Real.sqrt 2 * a * B.1 + b * B.2 = 1) ∧
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2)) →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 1/x^2 + 2/y^2 ≥ 4) ∧
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 1/x^2 + 2/y^2 = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1336_133696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_preserves_approx_newton_matches_continued_fraction_l1336_133600

/-- Continued fraction approximant for α or β -/
noncomputable def ContinuedFractionApprox (p q : ℝ) : ℕ → ℝ
| 0 => p
| (n+1) => p - q / ContinuedFractionApprox p q n

/-- Newton's method iteration -/
noncomputable def NewtonIteration (p q : ℝ) (x : ℝ) : ℝ :=
  (x^2 - q) / (2*x - p)

/-- Newton's method iterate -/
noncomputable def newton_iterate (p q : ℝ) : ℕ → ℝ → ℝ
| 0, x => x
| (n+1), x => newton_iterate p q n (NewtonIteration p q x)

/-- Theorem: Newton's method preserves continued fraction approximants -/
theorem newton_preserves_approx (p q : ℝ) (n : ℕ) :
  ∀ x, x = ContinuedFractionApprox p q n →
    NewtonIteration p q x = ContinuedFractionApprox p q (n+1) :=
by
  sorry

/-- Main theorem: If x_0 is a continued fraction approximant, all subsequent x_n are also approximants -/
theorem newton_matches_continued_fraction (p q : ℝ) :
  ∀ x₀ n, (∃ k, x₀ = ContinuedFractionApprox p q k) →
    (∃ m, (newton_iterate p q n x₀) = ContinuedFractionApprox p q m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_preserves_approx_newton_matches_continued_fraction_l1336_133600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_triangle_intersection_l1336_133654

-- Define necessary functions as axioms
axiom rectangle_centered_in_circle : ℝ → ℝ → ℝ → Prop
axiom triangle_inscribed_in_circle : ℝ → ℝ → ℝ → Prop
axiom triangle_vertex_at_circle_center : ℝ → ℝ → ℝ → Prop
axiom triangle_leg_aligns_with_rectangle_side : ℝ → ℝ → ℝ → ℝ → Prop
axiom area_of_intersection_not_covered_by_triangle : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ

theorem rectangle_circle_triangle_intersection
  (circle_radius : ℝ)
  (rectangle_length rectangle_width : ℝ)
  (triangle_leg1 triangle_leg2 : ℝ)
  (h_radius : circle_radius = 5)
  (h_rectangle : rectangle_length = 10 ∧ rectangle_width = 3)
  (h_triangle : triangle_leg1 = 6 ∧ triangle_leg2 = 8)
  (h_centered : rectangle_centered_in_circle circle_radius rectangle_length rectangle_width)
  (h_inscribed : triangle_inscribed_in_circle circle_radius triangle_leg1 triangle_leg2)
  (h_vertex : triangle_vertex_at_circle_center circle_radius triangle_leg1 triangle_leg2)
  (h_align : triangle_leg_aligns_with_rectangle_side rectangle_length rectangle_width triangle_leg1 triangle_leg2) :
  area_of_intersection_not_covered_by_triangle circle_radius rectangle_length rectangle_width triangle_leg1 triangle_leg2 = 6 * Real.pi - 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_triangle_intersection_l1336_133654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_officer_soldiers_count_l1336_133665

theorem officer_soldiers_count : ∃ (n : ℕ), 
  (∃ (s : ℕ), s^2 + 30 = n) ∧ 
  (∃ (s : ℕ), (s + 1)^2 = n + 50) ∧ 
  n = 1975 := by
  -- We'll use 'sorry' to skip the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_officer_soldiers_count_l1336_133665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l1336_133691

noncomputable def proj (a : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * u.1 + a.2 * u.2
  let magnitude_squared := a.1 * a.1 + a.2 * a.2
  (dot_product / magnitude_squared * a.1, dot_product / magnitude_squared * a.2)

theorem vector_satisfies_projections : 
  let u : ℝ × ℝ := (-1.6, 8.9)
  proj (3, 2) u = (6, 4) ∧ proj (1, 4) u = (2, 8) := by
  sorry

#check vector_satisfies_projections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l1336_133691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_graph_length_squared_l1336_133632

noncomputable def p (x : ℝ) : ℝ := -x + 1
noncomputable def q (x : ℝ) : ℝ := x + 1
noncomputable def r : ℝ → ℝ := λ _ => 3

noncomputable def n (x : ℝ) : ℝ := min (min (p x) (q x)) (r x)

noncomputable def graph_length_squared (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (b - a)^2 + (f b - f a)^2

theorem n_graph_length_squared :
  graph_length_squared n (-4) (-2) +
  graph_length_squared n (-2) 2 +
  graph_length_squared n 2 4 = 48 + 16 * Real.sqrt 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_graph_length_squared_l1336_133632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1336_133642

/-- Represents a parabola with equation x² = ay where a is a real number -/
structure Parabola where
  a : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (0, p.a / 4)

/-- Theorem: The focus of the parabola x² = ay is (0, a/4) -/
theorem parabola_focus (p : Parabola) : focus p = (0, p.a / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1336_133642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lit_area_approximation_l1336_133619

/-- The area of a quarter circle with radius 21 meters -/
noncomputable def lit_area : ℝ := (Real.pi * 21^2) / 4

/-- The side length of the square plot -/
def plot_side : ℝ := 50

/-- Theorem stating that the lit area is approximately 346.36 square meters -/
theorem lit_area_approximation : 
  346.35 < lit_area ∧ lit_area < 346.37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lit_area_approximation_l1336_133619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l1336_133651

open Set

-- Define the sets
def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 0}

-- State the theorem
theorem complement_of_union :
  (A ∪ B)ᶜ = {x : ℝ | x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l1336_133651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_l1336_133625

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- Theorem statement
theorem f_sum_reciprocal (x : ℝ) (hx : x ≠ 0) : f x + f (1/x) = 1 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp [hx]
  -- The proof is omitted for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_l1336_133625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l1336_133626

-- Define the ellipse parameters
noncomputable def major_axis_length : ℝ := 12
noncomputable def eccentricity : ℝ := 2/3

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the left vertex of the hyperbola
noncomputable def left_vertex : ℝ × ℝ := (-3, 0)

-- Theorem for the ellipse equation
theorem ellipse_equation : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = major_axis_length / 2 ∧
  (a^2 - b^2) / a^2 = eccentricity^2 ∧
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 36 + y^2 / 20 = 1) := by
  sorry

-- Theorem for the parabola equation
theorem parabola_equation :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x y : ℝ), y^2 = -2 * p * x ↔ y^2 = -12 * x) ∧
  (-p / 2, 0) = left_vertex := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l1336_133626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_l1336_133633

/-- Proves that given the conditions, b worked for 9 days -/
theorem b_work_days (a_days b_days c_days : ℕ) 
  (a_wage b_wage c_wage : ℕ) (total_earning : ℕ) :
  a_days = 6 →
  c_days = 4 →
  a_wage * 4 = b_wage * 3 →
  b_wage * 5 = c_wage * 4 →
  c_wage = 105 →
  total_earning = a_wage * a_days + b_wage * b_days + c_wage * c_days →
  total_earning = 1554 →
  b_days = 9 := by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

#check b_work_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_l1336_133633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_41_l1336_133639

theorem binomial_sum_equals_41 : (Nat.choose 7 4) + (Nat.choose 6 5) = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_41_l1336_133639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_AD_l1336_133660

-- Define the rectangle
noncomputable def rectangle_width : ℝ := 6
noncomputable def rectangle_height : ℝ := 3

-- Define points A, M, and P
noncomputable def A : ℝ × ℝ := (0, rectangle_height)
noncomputable def M : ℝ × ℝ := (rectangle_width / 2, 0)

-- Define the radii of the circles
noncomputable def circle_M_radius : ℝ := 1.5
noncomputable def circle_A_radius : ℝ := 5

-- Define P as a point satisfying both circle equations
noncomputable def P : ℝ × ℝ := sorry

-- State the theorem
theorem distance_P_to_AD : 
  let (x, y) := P
  (x - M.1)^2 + (y - M.2)^2 = circle_M_radius^2 ∧
  (x - A.1)^2 + (y - A.2)^2 = circle_A_radius^2 →
  x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_AD_l1336_133660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_values_l1336_133659

/-- Unit vector in the direction of positive x-axis -/
noncomputable def i : ℝ × ℝ := (1, 0)

/-- Unit vector in the direction of positive y-axis -/
noncomputable def j : ℝ × ℝ := (0, 1)

/-- Vector AB -/
noncomputable def AB : ℝ × ℝ := (4, 3)

/-- Vector AC parameterized by k -/
noncomputable def AC (k : ℝ) : ℝ × ℝ := (k, -1/2)

/-- Vector BC parameterized by k -/
noncomputable def BC (k : ℝ) : ℝ × ℝ := (k - 4, -7/2)

/-- Dot product of two 2D vectors -/
noncomputable def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Condition for right-angled triangle -/
def is_right_angled (k : ℝ) : Prop :=
  dot AB (BC k) = 0 ∨ dot AB (AC k) = 0 ∨ dot (AC k) (BC k) = 0

/-- The main theorem -/
theorem right_angled_triangle_values :
  ∃ (s : Finset ℝ), s.card = 4 ∧ (∀ k, is_right_angled k ↔ k ∈ s) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_values_l1336_133659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_hit_seven_l1336_133623

-- Define the set of players
inductive Player : Type
| Alex | Bobby | Carla | Diana | Eric | Fiona

-- Define the function that maps players to their scores
def score : Player → ℕ
| Player.Alex => 15
| Player.Bobby => 9
| Player.Carla => 13
| Player.Diana => 12
| Player.Eric => 14
| Player.Fiona => 17

-- Define the set of possible dart values
def DartValue := Fin 10

-- Define a function that returns the two dart values for each player
def darts : Player → (DartValue × DartValue) :=
  sorry -- Implementation not provided, as it's part of the proof

-- Define a predicate that checks if a player hit a specific number
def hit_number (p : Player) (n : DartValue) : Prop :=
  (darts p).1 = n ∨ (darts p).2 = n

-- Theorem stating that Diana hit the number 7
theorem diana_hit_seven :
  hit_number Player.Diana ⟨7, sorry⟩ ∧
  (∀ p : Player, p ≠ Player.Diana → ¬hit_number p ⟨7, sorry⟩) :=
sorry

#check diana_hit_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_hit_seven_l1336_133623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooters_probability_l1336_133618

/-- The number of shooters -/
def n : ℕ := 5

/-- The number of attempts -/
def m : ℕ := 3

/-- The probability of a single shooter hitting the target -/
noncomputable def p : ℚ := 2/3

/-- The probability that all shooters hit the target at least once in m attempts -/
noncomputable def prob_all_hit_at_least_once (n m : ℕ) (p : ℚ) : ℚ :=
  1 - (1 - p^n)^m

theorem shooters_probability :
  prob_all_hit_at_least_once n m p = 1 - (1 - (2/3)^5)^3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooters_probability_l1336_133618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_numbers_satisfy_conditions_l1336_133679

theorem no_numbers_satisfy_conditions : 
  ¬∃ n : ℕ+, (53 % n.val = 3) ∧ (4 ∣ n.val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_numbers_satisfy_conditions_l1336_133679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_transformation_l1336_133690

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Converts spherical coordinates to rectangular coordinates -/
noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : Point3D :=
  { x := ρ * Real.sin φ * Real.cos θ
    y := ρ * Real.sin φ * Real.sin θ
    z := ρ * Real.cos φ }

theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  let p₁ := sphericalToRectangular ρ θ φ
  let p₂ := sphericalToRectangular ρ (-θ) φ
  p₁ = ⟨-5, -7, 4⟩ → p₂ = ⟨-5, 7, 4⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_transformation_l1336_133690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_physical_activities_average_l1336_133653

theorem school_physical_activities_average (s : ℕ) : 
  let sixth_grade_count := 3 * s
  let seventh_grade_count := s
  let eighth_grade_count := s / 2
  let sixth_grade_minutes := 18
  let seventh_grade_minutes := 20
  let seventh_grade_stretch := 5
  let eighth_grade_minutes := 12
  let total_minutes := sixth_grade_count * sixth_grade_minutes + 
                       seventh_grade_count * (seventh_grade_minutes + seventh_grade_stretch) + 
                       eighth_grade_count * eighth_grade_minutes
  let total_students := sixth_grade_count + seventh_grade_count + eighth_grade_count
  (total_minutes : ℚ) / total_students = 170 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_physical_activities_average_l1336_133653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_plus_mu_constant_l1336_133622

/-- Parabola C₁ -/
def C₁ (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of C₁ -/
def F : ℝ × ℝ := (1, 0)

/-- Point on parabola C₁ -/
def M : ℝ × ℝ := (3, 2)

/-- Distance from M to F is 4 -/
axiom dist_M_F : Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 4

/-- Ellipse C₂ -/
def C₂ (x y : ℝ) : Prop := y^2/2 + x^2 = 1

/-- C₂ passes through F -/
axiom C₂_passes_F : C₂ F.1 F.2

/-- Line l passing through F -/
def l (k : ℝ) (x : ℝ) : ℝ := k*(x - F.1)

/-- Intersection points of l with C₁ -/
noncomputable def A (k : ℝ) : ℝ × ℝ := sorry
noncomputable def B (k : ℝ) : ℝ × ℝ := sorry

/-- Intersection of l with y-axis -/
def N (k : ℝ) : ℝ × ℝ := (0, -k)

/-- Definition of λ and μ -/
noncomputable def lambda (k : ℝ) : ℝ := (A k).1 / (1 - (A k).1)
noncomputable def mu (k : ℝ) : ℝ := (B k).1 / (1 - (B k).1)

/-- Main theorem -/
theorem lambda_plus_mu_constant (k : ℝ) : lambda k + mu k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_plus_mu_constant_l1336_133622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l1336_133669

/-- A polynomial is in difference of squares form if it can be written as x^2 - y^2 for some x and y -/
def is_difference_of_squares (p : Polynomial ℤ) : Prop :=
  ∃ x y : Polynomial ℤ, p = x^2 - y^2

theorem difference_of_squares_factorization :
  (∃ a b : ℤ, is_difference_of_squares (Polynomial.C a^2 - Polynomial.C b^2)) ∧
  (∃ b : ℤ, is_difference_of_squares (16 * Polynomial.X^2 - Polynomial.C b^2)) ∧
  (∃ b : ℤ, is_difference_of_squares (-(Polynomial.X^2) + 25 * Polynomial.C b^2)) ∧
  ¬is_difference_of_squares (Polynomial.C (-4) - Polynomial.X^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l1336_133669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_problem_l1336_133603

theorem selection_problem (n_boys n_girls n_select : ℕ) 
  (h1 : n_boys = 4) (h2 : n_girls = 3) (h3 : n_select = 4) :
  (Finset.sum (Finset.range (n_girls + 1)) (λ k ↦ 
    Nat.choose n_girls k * Nat.choose n_boys (n_select - k))) - 
  Nat.choose n_boys n_select = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_problem_l1336_133603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_plane_l1336_133621

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the line and plane
variable (m : Submodule ℝ V) (α : Submodule ℝ V)

-- Define the conditions
variable (h1 : FiniteDimensional.finrank ℝ m = 1)
variable (h2 : FiniteDimensional.finrank ℝ α = FiniteDimensional.finrank ℝ V - 1)
variable (h3 : m ⊓ α ≠ ⊥)
variable (h4 : m.orthogonal ≠ α)

-- State the theorem
theorem unique_perpendicular_plane :
  ∃! P : Submodule ℝ V, m ≤ P ∧ 
    FiniteDimensional.finrank ℝ P = FiniteDimensional.finrank ℝ V - 1 ∧ 
    P.orthogonal = α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_plane_l1336_133621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_children_count_l1336_133686

/-- The number of children in the school -/
def C : ℕ := sorry

/-- The number of absent children -/
def absent : ℕ := 370

/-- The number of bananas each child was originally supposed to get -/
def original_bananas : ℕ := 2

/-- The number of bananas each present child actually got -/
def actual_bananas : ℕ := 4

theorem school_children_count :
  (C * original_bananas = (C - absent) * actual_bananas) →
  C = 740 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_children_count_l1336_133686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1336_133699

/-- The curve in polar coordinates -/
noncomputable def curve (θ : ℝ) : ℝ := Real.cos θ + Real.sin θ

/-- The line in parametric form -/
noncomputable def line (t : ℝ) : ℝ × ℝ := (1/2 - Real.sqrt 2/2 * t, Real.sqrt 2/2 * t)

/-- The theorem stating the distance between intersection points -/
theorem intersection_distance : 
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
  let (x₁, y₁) := line t₁
  let (x₂, y₂) := line t₂
  x₁^2 + y₁^2 = curve (Real.arctan (y₁ / x₁)) ∧
  x₂^2 + y₂^2 = curve (Real.arctan (y₂ / x₂)) ∧
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 6 / 2 := by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1336_133699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_l1336_133646

/-- Given a principal amount and an interest rate, 
    if the amount after 2 years is 2420 and after 3 years is 2783, 
    then the interest rate is approximately 15% -/
theorem compound_interest_rate (P r : ℝ) 
  (h1 : P * (1 + r / 100)^2 = 2420)
  (h2 : P * (1 + r / 100)^3 = 2783) :
  ∃ ε > 0, |r - 15| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_l1336_133646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_chapter_pages_l1336_133629

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The book in the problem -/
def problem_book : Book :=
  { chapter1_pages := 37,
    chapter2_pages := 37 + 43 }

theorem second_chapter_pages : problem_book.chapter2_pages = 80 := by
  rfl

#eval problem_book.chapter2_pages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_chapter_pages_l1336_133629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_solution_set_part1_alt_range_of_a_part2_alt_l1336_133698

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + 4| + x - 2*a + a^2

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 ≥ 6} = Set.Iic (-10) ∪ Set.Ici (2/3) := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f x a ≥ 10 - |2 - x|} = Set.Iic (-2) ∪ Set.Ici 4 := by sorry

-- Additional definitions to make the statement more precise
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f x a ≥ 6}
def satisfies_inequality (a : ℝ) : Prop := ∀ x, f x a ≥ 10 - |2 - x|

-- Restating the theorems using the additional definitions
theorem solution_set_part1_alt :
  solution_set 2 = Set.Iic (-10) ∪ Set.Ici (2/3) := by sorry

theorem range_of_a_part2_alt :
  {a : ℝ | satisfies_inequality a} = Set.Iic (-2) ∪ Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_solution_set_part1_alt_range_of_a_part2_alt_l1336_133698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1336_133605

/-- The solution set of the inequality 1 + √3 tan x ≥ 0 -/
noncomputable def solution_set : Set ℝ :=
  {x | ∃ k : ℤ, -Real.pi/6 + k*Real.pi ≤ x ∧ x < Real.pi/2 + k*Real.pi}

/-- Theorem stating that the solution_set is correct for the given inequality -/
theorem solution_set_correct :
  ∀ x : ℝ, (1 + Real.sqrt 3 * Real.tan x ≥ 0) ↔ x ∈ solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1336_133605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1336_133630

theorem negation_of_universal_proposition (f : ℕ → ℕ) :
  (¬ ∀ n : ℕ, f n ≤ n) ↔ (∃ n : ℕ, f n > n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1336_133630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_correct_l1336_133644

/-- Represents a coin that may be counterfeit -/
inductive Coin
| genuine : Coin
| lighter : Coin
| heavier : Coin

/-- The minimum number of weighings needed to identify a counterfeit coin among n coins -/
noncomputable def min_weighings (n : ℕ) : ℕ := 
  Int.toNat ⌈Real.log ↑n / Real.log 3⌉

/-- A balance scale comparison result -/
inductive Comparison
| equal : Comparison
| left_heavier : Comparison
| right_heavier : Comparison

/-- Represents a weighing operation on the balance scale -/
def weighing (left : List Coin) (right : List Coin) : Comparison :=
  sorry

theorem min_weighings_correct (n : ℕ) :
  ∀ (coins : List Coin),
    coins.length = n + 1 →
    (∃ (i : Fin coins.length), (coins.get i = Coin.lighter ∨ coins.get i = Coin.heavier)) →
    (∀ (i j : Fin coins.length), i ≠ j → coins.get i = Coin.genuine ∨ coins.get j = Coin.genuine) →
    ∃ (k : ℕ),
      k ≤ min_weighings n ∧
      ∀ (algorithm : ℕ → List Coin → List Coin → Comparison),
        ∃ (result : List (List Coin × List Coin)),
          result.length = k ∧
          (∀ (step : Fin result.length),
            algorithm step.val (result.get step).fst (result.get step).snd =
            weighing (result.get step).fst (result.get step).snd) ∧
          ∃ (i : Fin coins.length),
            (coins.get i = Coin.lighter ∨ coins.get i = Coin.heavier) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_correct_l1336_133644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_overtake_theorem_l1336_133640

/-- Calculates the time it takes for a faster plane to overtake a slower plane -/
noncomputable def overtake_time (speed_a speed_b headwind crosswind : ℝ) (takeoff_diff : ℝ) : ℝ :=
  let ground_speed_a := speed_a - headwind
  let ground_speed_b := speed_b - crosswind
  let t := (ground_speed_a * takeoff_diff) / (ground_speed_b - ground_speed_a)
  t - takeoff_diff

theorem plane_overtake_theorem (speed_a speed_b headwind crosswind takeoff_diff : ℝ) 
  (h1 : speed_a = 200)
  (h2 : speed_b = 300)
  (h3 : headwind = 15)
  (h4 : crosswind = 10)
  (h5 : takeoff_diff = 40 / 60) :
  ∃ (result : ℝ), abs (overtake_time speed_a speed_b headwind crosswind takeoff_diff - result) < 0.01 ∧ 
  30.28 < result ∧ result < 30.30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_overtake_theorem_l1336_133640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_determined_by_initial_state_l1336_133613

/-- Represents the state of the game -/
inductive GameState
  | Big
  | Small

/-- Represents a player -/
inductive Player
  | Anna
  | Beatrice

/-- The game setup -/
structure GameSetup where
  n : Nat
  deck : List Nat
  h_deck_size : deck.length = n
  h_deck_range : ∀ i ∈ deck, 1 ≤ i ∧ i ≤ n

/-- The game state after a move -/
structure GameMove where
  k : Nat
  top_k_cards : List Nat
  h_top_k_size : top_k_cards.length = k
  h_top_k_range : ∀ i ∈ top_k_cards, 1 ≤ i ∧ i ≤ k

/-- Determines if a game state is "big" or "small" -/
def determineGameState (move : GameMove) : GameState :=
  if move.k = (move.top_k_cards.minimum?).getD 0 then GameState.Small else GameState.Big

/-- Determines the winner based on the initial game state -/
def determineWinner (initialState : GameState) : Player :=
  match initialState with
  | GameState.Big => Player.Anna
  | GameState.Small => Player.Beatrice

/-- Main theorem: The winner is determined by the initial game state -/
theorem winner_determined_by_initial_state
  (setup : GameSetup)
  (initial_move : GameMove)
  (h_initial_move : initial_move.top_k_cards = setup.deck.take initial_move.k) :
  determineWinner (determineGameState initial_move) = 
    if initial_move.k ≠ (initial_move.top_k_cards.minimum?).getD 0
    then Player.Anna
    else Player.Beatrice := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_determined_by_initial_state_l1336_133613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_printable_example_l1336_133673

/-- The number of pages that can be printed given the cost per page and available funds -/
def pages_printable (cost_per_page : ℚ) (available_funds : ℚ) : ℕ :=
  (available_funds * 100 / cost_per_page).floor.toNat

/-- Theorem: Given a printing cost of 3 cents per page and $15 available, 
    the number of pages that can be printed is equal to 500 -/
theorem pages_printable_example : pages_printable (3 / 100) 15 = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_printable_example_l1336_133673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_in_centers_l1336_133697

/-- Represents a position in the infinite grid -/
structure Position where
  x : Int
  y : Int

/-- The spiral arrangement of natural numbers on the grid -/
def spiral : Position → Nat := sorry

/-- The sum of numbers at the vertices of a square -/
def squareSum (p : Position) : Nat := sorry

/-- Theorem: For any positive integer n, there are infinitely many squares
    whose center contains a multiple of n -/
theorem infinitely_many_multiples_in_centers (n : Nat) (hn : n > 0) :
  ∀ k : Nat, ∃ p : Position, k ≤ (squareSum p) ∧ n ∣ squareSum p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_in_centers_l1336_133697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diego_half_block_time_l1336_133638

/-- Proves that Diego's time to run half the block is 5 minutes given the conditions -/
theorem diego_half_block_time (carlos_time : ℝ) (average_time : ℝ) : 
  2 * average_time - carlos_time = 5 := by
  have h1 : carlos_time = 3 := by sorry
  have h2 : average_time = 4 := by sorry  -- 240 seconds = 4 minutes
  calc
    2 * average_time - carlos_time = 2 * 4 - 3 := by rw [h1, h2]
    _ = 8 - 3 := by norm_num
    _ = 5 := by norm_num

#eval (2 : ℝ) * 4 - 3  -- This should output 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diego_half_block_time_l1336_133638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_value_l1336_133666

-- Define the function f(x) as noncomputable due to its dependency on Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a - 3 * x)

-- State the theorem
theorem domain_implies_a_value (a : ℝ) :
  (∀ x < 2, f a x ∈ Set.range Real.log) →
  a = 6 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_value_l1336_133666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l1336_133650

/-- Predicate for a set of points forming a square in ℝ² -/
def is_square (S : Set (ℝ × ℝ)) : Prop := sorry

/-- Calculate the area of a set of points in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Get the midpoints of the sides of a square -/
def midpoints_of_square (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Construct a square from the diagonals connecting opposite corners to midpoints -/
def square_from_diagonals (S : Set (ℝ × ℝ)) (midpoints : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The area of the inner square formed by connecting the midpoints of each side
    to the opposite corners of a square with area 100 square units is 25 square units. -/
theorem inner_square_area : ∀ (S : Set (ℝ × ℝ)),
  is_square S →
  area S = 100 →
  area (square_from_diagonals S (midpoints_of_square S)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l1336_133650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_36288_l1336_133683

/-- The number of positive factors of 36288 -/
noncomputable def num_factors_36288 : ℕ :=
  Nat.card {d : ℕ | d > 0 ∧ 36288 % d = 0}

/-- Theorem stating that 36288 has 70 positive factors -/
theorem factors_of_36288 : num_factors_36288 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_36288_l1336_133683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_q_is_false_l1336_133677

open Real

def p : Prop := ∀ x : ℝ, 2 * x^2 + 2 * x + (1/2) < 0

def q : Prop := ∃ x : ℝ, sin x - cos x = Real.sqrt 2

theorem not_q_is_false : ¬(¬q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_q_is_false_l1336_133677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_theorem_l1336_133671

/-- The area enclosed between an inscribed circle in an equilateral triangle
    and a circle drawn from a vertex of the triangle. -/
noncomputable def area_between_circles (a : ℝ) : ℝ :=
  (a^2 / 72) * (4 * Real.pi - 3 * Real.sqrt 3)

/-- Theorem: The area enclosed between an inscribed circle in an equilateral triangle
    with side length a and a circle with radius 0.5a drawn from a vertex of the triangle
    is equal to (a^2 / 72) * (4π - 3√3). -/
theorem area_between_circles_theorem (a : ℝ) (h : a > 0) :
  let triangle_side := a
  let inscribed_circle_radius := (a * Real.sqrt 3) / 6
  let vertex_circle_radius := a / 2
  area_between_circles a =
    (Real.pi * vertex_circle_radius^2 / 6) -
    (1 / 3) * ((Real.sqrt 3 / 4 * triangle_side^2) - (Real.pi * inscribed_circle_radius^2)) :=
by
  sorry

#check area_between_circles_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_theorem_l1336_133671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_27_l1336_133647

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its four vertices -/
noncomputable def trapezoidArea (e f g h : Point2D) : ℝ :=
  let height := g.x - e.x
  let base1 := f.y - e.y
  let base2 := g.y - h.y
  (base1 + base2) * height / 2

/-- Theorem stating that the area of the given trapezoid is 27 square units -/
theorem trapezoid_area_is_27 :
  let e := Point2D.mk 2 0
  let f := Point2D.mk 2 3
  let g := Point2D.mk 8 5
  let h := Point2D.mk 8 (-1)
  trapezoidArea e f g h = 27 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_27_l1336_133647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_pyramid_l1336_133678

/-- Represents a triangular pyramid --/
structure TriangularPyramid where
  height : ℝ
  lateral_height : ℝ
  base_perimeter : ℝ

/-- Checks if a triangular pyramid with given dimensions can exist --/
def is_valid_pyramid (p : TriangularPyramid) : Prop :=
  let s := p.base_perimeter / 2
  let r := s / 3  -- Assuming equilateral base triangle
  let area_triangle := r * s
  area_triangle > Real.pi * r^2

/-- Theorem stating that a triangular pyramid with the given dimensions cannot exist --/
theorem no_such_pyramid :
  ¬ ∃ (p : TriangularPyramid), p.height = 60 ∧ p.lateral_height = 61 ∧ p.base_perimeter = 62 ∧ is_valid_pyramid p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_pyramid_l1336_133678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_probability_l1336_133645

/-- A regular octahedron -/
structure Octahedron :=
  (faces : Fin 8 → ℕ)
  (distinct : ∀ i j, i ≠ j → faces i ≠ faces j)

/-- The set of numbers to be placed on the octahedron -/
def OctahedronNumbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 9}

/-- Two numbers are consecutive if they differ by 1 or are 1 and 9 -/
def consecutive (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1) ∨ (a + 1 = b) ∨ (b + 1 = a)

/-- Adjacent faces on the octahedron -/
def adjacent : Fin 8 → Fin 8 → Prop := sorry

/-- A valid configuration of numbers on the octahedron -/
def valid_configuration (o : Octahedron) : Prop :=
  (∀ i, o.faces i ∈ OctahedronNumbers) ∧
  (∀ i j, adjacent i j → ¬consecutive (o.faces i) (o.faces j))

/-- Assume Fintype instances for the required types -/
instance : Fintype Octahedron := sorry

instance : Fintype {o : Octahedron // valid_configuration o} := sorry

instance : Fintype {o : Octahedron // ∀ i, o.faces i ∈ OctahedronNumbers} := sorry

/-- The main theorem -/
theorem octahedron_probability :
  (Fintype.card {o : Octahedron // valid_configuration o} : ℚ) /
  (Fintype.card {o : Octahedron // ∀ i, o.faces i ∈ OctahedronNumbers} : ℚ) = 1 / 84 := by
  sorry

#check octahedron_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_probability_l1336_133645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l1336_133693

/-- Rounds a number to the nearest multiple of 5, rounding 2.5 up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

/-- The sum of integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The sum of integers from 1 to n, each rounded to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => roundToNearestFive (i + 1))

theorem sum_equals_rounded_sum :
  sumToN 200 = sumRoundedToN 200 := by
  sorry

#eval sumToN 200
#eval sumRoundedToN 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l1336_133693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_uniformly_continuous_sin_uniformly_continuous_l1336_133628

-- Define uniform continuity
def uniformly_continuous (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → |x₁ - x₂| < δ → |f x₁ - f x₂| < ε

-- Theorem for √x on [1, +∞)
theorem sqrt_uniformly_continuous :
  uniformly_continuous (λ x => Real.sqrt x) (Set.Ici 1) := by
  sorry

-- Theorem for sin x on (-∞, +∞)
theorem sin_uniformly_continuous :
  uniformly_continuous Real.sin Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_uniformly_continuous_sin_uniformly_continuous_l1336_133628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_large_subset_l1336_133687

theorem divisibility_in_large_subset (S : Finset ℕ) : 
  S.card = 51 → (∀ n ∈ S, 1 ≤ n ∧ n ≤ 100) → 
  ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x ∣ y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_large_subset_l1336_133687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_four_equals_one_l1336_133608

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b / x + 3

-- State the theorem
theorem f_negative_four_equals_one (a b : ℝ) :
  f a b 4 = 5 → f a b (-4) = 1 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_four_equals_one_l1336_133608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l1336_133655

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := ((x + 5) / 5) ^ (1/4)

-- State the theorem
theorem h_equality :
  ∃ x : ℝ, h (3 * x) = 3 * h x ∧ x = -200/39 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equality_l1336_133655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_line_intersection_proof_l1336_133675

/-- The value of a for which the line y = √2 x is tangent to the circle (x - a)² + y² = 2 -/
noncomputable def tangent_circle_line_intersection : ℝ := Real.sqrt 3

theorem tangent_circle_line_intersection_proof :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, y = Real.sqrt 2 * x → (x - a)^2 + y^2 = 2 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, y' = Real.sqrt 2 * x' → 
      |((x' - a)^2 + y'^2 - 2)| < δ → 
      ((x - x')^2 + (y - y')^2).sqrt < ε) →
  a = tangent_circle_line_intersection :=
by
  sorry

#check tangent_circle_line_intersection_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_line_intersection_proof_l1336_133675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1336_133682

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x) - 1

-- Theorem statement
theorem f_properties :
  (∃ a b : ℝ, ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y) ∧
  (∃ c : ℝ, ∀ x : ℝ, f (c - x) = f (c + x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1336_133682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_intersection_slope_range_l1336_133692

/-- Given points A(1, 1) and B(-2, 3), if the line y = ax - 1 intersects 
    the line segment AB, then the range of a is (-∞, -2] ∪ [2, +∞). -/
theorem line_segment_intersection_slope_range :
  let A : ℝ × ℝ := (1, 1)
  let B : ℝ × ℝ := (-2, 3)
  let line_intersects_segment (a : ℝ) : Prop := 
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    (1 - t) * A.fst + t * B.fst = ((1 - t) * A.snd + t * B.snd + 1) / a
  {a : ℝ | line_intersects_segment a} = 
    Set.Iic (-2 : ℝ) ∪ Set.Ici (2 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_intersection_slope_range_l1336_133692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l1336_133631

/-- Calculates the number of bricks required to pave a rectangular courtyard -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℕ :=
  let courtyard_area := courtyard_length * courtyard_width
  let brick_area := brick_length * brick_width / 10000  -- Convert cm² to m²
  (courtyard_area / brick_area).ceil.toNat

/-- Proves that 20,000 bricks are required for the given courtyard and brick dimensions -/
theorem courtyard_paving :
  bricks_required 25 16 20 10 = 20000 := by
  sorry

#eval bricks_required 25 16 20 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l1336_133631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l1336_133662

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_properties
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a (-2) = 9) :
  (∃ g : ℝ → ℝ, g = f (1/3)) ∧
  (∀ m : ℝ, f (1/3) (2*m - 1) - f (1/3) (m + 3) < 0 ↔ m > 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l1336_133662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l1336_133663

/-- The number of vertices in the regular polygon -/
def n : ℕ := 20

/-- The exponent used in the problem -/
def m : ℕ := 1995

/-- A complex number representing a vertex of the regular n-gon inscribed in the unit circle -/
noncomputable def vertex (k : ℕ) : ℂ := Complex.exp (2 * Real.pi * k * Complex.I / n)

/-- The set of all vertices of the regular n-gon -/
noncomputable def vertices : Finset ℂ := Finset.image vertex (Finset.range n)

/-- The set of complex numbers obtained by raising each vertex to the m-th power -/
noncomputable def raised_vertices : Finset ℂ := Finset.image (λ z => z ^ m) vertices

/-- The theorem stating that the number of distinct points is 4 -/
theorem distinct_points_count : Finset.card raised_vertices = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l1336_133663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l1336_133637

theorem fraction_irreducible (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l1336_133637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lovely_couple_divisor_mod_six_infinitely_many_lovely_couples_l1336_133604

/-- A lovely couple is a pair of positive integers (n, k) with k > 1 such that
    there exists an n × n table of ones and zeros where:
    - In every row, there are exactly k ones.
    - For each pair of rows, there is exactly one column such that at both
      intersections of that column with the mentioned rows, the number one is written. -/
structure LovelyCouple (n k : ℕ) : Prop where
  n_pos : n > 0
  k_gt_one : k > 1
  lovely : ∃ (table : Fin n → Fin n → Bool),
    (∀ i, (Finset.filter (λ j ↦ table i j) Finset.univ).card = k) ∧
    (∀ i₁ i₂, i₁ ≠ i₂ → ∃! j, table i₁ j ∧ table i₂ j)

/-- For any lovely couple (n, k), if d is a divisor of n = k^2 - k + 1 and d ≠ 1,
    then d ≡ 1 or 3 (mod 6) -/
theorem lovely_couple_divisor_mod_six (n k : ℕ) (h : LovelyCouple n k) :
  ∀ d : ℕ, d > 1 → d ∣ (k^2 - k + 1) → d % 6 = 1 ∨ d % 6 = 3 := by
  sorry

/-- There exist infinitely many lovely couples -/
theorem infinitely_many_lovely_couples :
  ∀ m : ℕ, ∃ n k : ℕ, n > m ∧ LovelyCouple n k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lovely_couple_divisor_mod_six_infinitely_many_lovely_couples_l1336_133604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1336_133601

/-- Represents a truncated pyramid with rectangular bases -/
structure TruncatedPyramid where
  m : ℝ  -- height
  a : ℝ  -- length of larger base
  b : ℝ  -- width of larger base
  c : ℝ  -- length of smaller base
  d : ℝ  -- width of smaller base
  h_positive : m > 0
  h_a_positive : a > 0
  h_b_positive : b > 0
  h_c_positive : c > 0
  h_d_positive : d > 0
  h_a_ge_c : a ≥ c
  h_b_ge_d : b ≥ d

/-- The volume of a truncated pyramid -/
noncomputable def volume (p : TruncatedPyramid) : ℝ :=
  (p.m / 6) * ((2 * p.a + p.c) * p.b + (2 * p.c + p.a) * p.d)

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula_correct (p : TruncatedPyramid) :
  volume p = (p.m / 3) * (p.a * p.b + Real.sqrt (p.a * p.b * p.c * p.d) + p.c * p.d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1336_133601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1336_133658

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, 2^x₀ - 2 ≤ a^2 - 3*a) → 
  (a ∈ Set.Icc 1 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1336_133658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l1336_133610

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log (x^2 + 1)) / (x + 4)

-- Define symmetry condition
def symmetric_about_zero (f : ℝ → ℝ) : Prop :=
  ∀ x, f (3 - x) = f (3 + x)

-- Theorem statement
theorem symmetry_of_f :
  symmetric_about_zero f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l1336_133610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_neg_one_l1336_133661

-- Define the inverse function as noncomputable
noncomputable def f_inv (x : ℝ) : ℝ := 2^(x + 1)

-- State the theorem
theorem f_of_one_eq_neg_one (f : ℝ → ℝ) (h : ∀ x, f_inv x = f⁻¹ x) : f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_neg_one_l1336_133661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_pole_distance_approximation_l1336_133612

def total_path_length : ℝ := 900
def bridge_length : ℝ := 42
def total_fence_poles : ℕ := 286

theorem fence_pole_distance_approximation :
  let fenced_length : ℝ := total_path_length - bridge_length
  let intervals_one_side : ℕ := total_fence_poles / 2 - 1
  let exact_distance : ℝ := fenced_length / (intervals_one_side : ℝ)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |exact_distance - 6| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_pole_distance_approximation_l1336_133612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l1336_133695

/-- Calculates the simple interest rate given the principal, amount, and time -/
noncomputable def simple_interest_rate (principal amount : ℝ) (time : ℝ) : ℝ :=
  (amount - principal) * 100 / (principal * time)

/-- The simple interest rate for the given problem -/
noncomputable def problem_rate : ℝ :=
  simple_interest_rate 1750 2000 2

theorem interest_rate_approximation :
  |problem_rate - 7.14| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l1336_133695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1336_133667

theorem unique_solution : ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (6 : ℚ) / m + (3 : ℚ) / n + 1 / (m * n : ℚ) = 1 ∧ m = 7 ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1336_133667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l1336_133611

theorem scientific_notation_equivalence :
  let original_number : ℝ := 0.000156
  let scientific_notation : ℝ := 1.56 * (10 ^ (-4 : ℤ))
  original_number = scientific_notation :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l1336_133611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1336_133680

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ :=
  (1 + Real.cos (2 * x)) / (4 * Real.sin (Real.pi / 2 + x)) - a * Real.sin (x / 2) * Real.cos (Real.pi - x / 2)

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∃ (M : ℝ), M = 2 ∧ ∀ x, f x a ≤ M) →
  a = Real.sqrt 15 ∨ a = -Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1336_133680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_combinable_with_sqrt_two_l1336_133664

theorem sqrt_combinable_with_sqrt_two :
  ∃ (a : ℚ), (Real.sqrt 8 : ℝ) = a * Real.sqrt 2 ∧
  (∀ (b : ℚ), (Real.sqrt 4 : ℝ) ≠ b * Real.sqrt 2) ∧
  (∀ (c : ℚ), (Real.sqrt 24 : ℝ) ≠ c * Real.sqrt 2) ∧
  (∀ (d : ℚ), (Real.sqrt 12 : ℝ) ≠ d * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_combinable_with_sqrt_two_l1336_133664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l1336_133602

theorem proposition_analysis : 
  (¬(∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2)) ∧ 
  (∀ x : ℝ, x^2 + x + 1 > 0) ∧ 
  ¬((∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2) ∧ (∀ x : ℝ, x^2 + x + 1 > 0)) ∧
  ¬((∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2) ∨ ¬(∀ x : ℝ, x^2 + x + 1 > 0)) ∧
  ((¬(∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2)) ∨ (∀ x : ℝ, x^2 + x + 1 > 0)) ∧
  ¬((¬(∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2)) ∨ (¬(∀ x : ℝ, x^2 + x + 1 > 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l1336_133602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_net_gain_is_155_l1336_133636

/-- Represents an item with its cost price and profit/loss percentage -/
structure Item where
  costPrice : ℕ
  profitLossPercentage : Int
  deriving Repr

/-- Calculates the selling price of an item -/
def sellingPrice (item : Item) : Int :=
  item.costPrice + item.costPrice * item.profitLossPercentage / 100

/-- Theorem: The total net gain from selling all items is 155 -/
theorem total_net_gain_is_155 (items : List Item) 
  (h1 : items = [
    { costPrice := 1200, profitLossPercentage := -15 },
    { costPrice := 2500, profitLossPercentage := 10 },
    { costPrice := 3300, profitLossPercentage := -5 },
    { costPrice := 4000, profitLossPercentage := 20 },
    { costPrice := 5500, profitLossPercentage := -10 }
  ]) : 
  (items.map sellingPrice).sum - (items.map (·.costPrice)).sum = 155 := by
  sorry

/-- Example calculation -/
def exampleItems : List Item := [
  { costPrice := 1200, profitLossPercentage := -15 },
  { costPrice := 2500, profitLossPercentage := 10 },
  { costPrice := 3300, profitLossPercentage := -5 },
  { costPrice := 4000, profitLossPercentage := 20 },
  { costPrice := 5500, profitLossPercentage := -10 }
]

#eval (exampleItems.map sellingPrice).sum - (exampleItems.map (·.costPrice)).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_net_gain_is_155_l1336_133636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mileage_l1336_133689

/-- Calculates the average gas mileage for a round trip with different vehicle efficiencies -/
noncomputable def averageGasMileage (totalDistance : ℝ) (efficiency1 : ℝ) (efficiency2 : ℝ) : ℝ :=
  totalDistance / ((totalDistance / 2 / efficiency1) + (totalDistance / 2 / efficiency2))

/-- Theorem stating that the average gas mileage for the given round trip is 24 mpg -/
theorem round_trip_mileage :
  averageGasMileage 240 30 20 = 24 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mileage_l1336_133689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_min_value_l1336_133616

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_odd_and_min_value (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) →
  (a = 1/2 ∧ ∀ x ∈ Set.Icc 1 5, f (1/2) x ≥ 1/6 ∧ f (1/2) 1 = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_min_value_l1336_133616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_circles_range_l1336_133609

/-- Two circles with exactly three common tangents -/
structure ThreeTangentCircles where
  a : ℝ
  b : ℝ
  C₁ : Set (ℝ × ℝ) := {(x, y) | (x - a)^2 + y^2 = 1}
  C₂ : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 - 2*b*y + b^2 - 4 = 0}
  three_tangents : a^2 + b^2 = 9 -- This condition replaces HasExactlyThreeCommonTangents

/-- The range of a² + b² - 6a - 8b for circles with three common tangents -/
theorem three_tangent_circles_range (c : ThreeTangentCircles) :
  -21 ≤ c.a^2 + c.b^2 - 6*c.a - 8*c.b ∧ c.a^2 + c.b^2 - 6*c.a - 8*c.b ≤ 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_circles_range_l1336_133609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1336_133641

open Real

theorem trig_identity (A : ℝ) : 
  (2 - (cos A / sin A) + (1 / sin A)) * (3 - (sin A / cos A) - (1 / cos A)) = 
  7 * sin A * cos A - 2 * (cos A)^2 - 3 * (sin A)^2 - 3 * cos A + sin A + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1336_133641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_350_meters_l1336_133684

-- Define the given parameters
noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmh : ℝ := 72
noncomputable def crossing_time : ℝ := 30

-- Define the function to convert km/h to m/s
noncomputable def km_per_hour_to_meter_per_second (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

-- Define the theorem
theorem bridge_length_is_350_meters :
  let train_speed_ms := km_per_hour_to_meter_per_second train_speed_kmh
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 350 := by
  -- Unfold the definitions
  unfold km_per_hour_to_meter_per_second
  unfold train_speed_kmh
  unfold crossing_time
  unfold train_length
  -- Simplify the expressions
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_350_meters_l1336_133684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_coefficient_l1336_133648

/-- A power function with coefficient (m^2 - 2m - 2) and exponent (2-m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^(2-m)

/-- Theorem stating that the only positive value of m that makes the coefficient of f equal to 1 is 3 -/
theorem power_function_coefficient (m : ℝ) (h : m > 0) :
  (∃ c : ℝ, ∀ x : ℝ, f m x = c * x^(2-m)) ↔ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_coefficient_l1336_133648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_equal_angles_circle_with_OP_diameter_l1336_133635

-- Define the curve C and line l
def C (x y : ℝ) : Prop := x^2 = 6*y
def l (x y k : ℝ) : Prop := y = k*x + 3

-- Define the intersection points M and N
def intersection_points (k : ℝ) : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ k ∧ l x₂ y₂ k :=
  sorry

-- Define the area of triangle MON
noncomputable def area_MON (k : ℝ) : ℝ := 9 * Real.sqrt (k^2 + 2)

-- Theorem for the area range
theorem area_range (k : ℝ) (h : 1 < k ∧ k < 2) : 
  27 < area_MON k ∧ area_MON k < 54 :=
  sorry

-- Define point P
def P : ℝ × ℝ := (0, -3)

-- Define angles POM and PON
noncomputable def angle_POM (k : ℝ) : ℝ := sorry
noncomputable def angle_PON (k : ℝ) : ℝ := sorry

-- Theorem for equal angles
theorem equal_angles (k : ℝ) : angle_POM k = angle_PON k := by
  sorry

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y+3)^2 = 36

-- Theorem for the circle equation
theorem circle_with_OP_diameter : 
  ∀ (x y : ℝ), (x - 0)^2 + (y - P.2)^2 = P.1^2 + P.2^2 ↔ circle_equation x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_range_equal_angles_circle_with_OP_diameter_l1336_133635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1336_133649

noncomputable def f (x : ℝ) := 1 / Real.log (3 * x + 1)

theorem f_domain : 
  {x : ℝ | f x ∈ Set.univ} = Set.Ioo (-1/3) 0 ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1336_133649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1336_133681

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (2, 0)

-- Define the function to calculate the area of a triangle
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem statement
theorem area_of_triangle_ABC :
  triangleArea A B C = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1336_133681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1336_133607

/-- The line on which point M lies -/
def line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

/-- The circle M -/
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

/-- Theorem stating that the given equation represents circle M -/
theorem circle_equation :
  ∃ (mx my : ℝ),
    line mx my ∧
    circle_M 3 0 ∧
    circle_M 0 1 ∧
    ∀ (x y : ℝ), circle_M x y ↔ (x - 1)^2 + (y + 1)^2 = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1336_133607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_with_inscribed_circles_l1336_133606

theorem square_area_with_inscribed_circles 
  (radius : ℝ) (h : radius = 1) : 
  ∃ (area : ℝ), area = 3 + Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_with_inscribed_circles_l1336_133606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_l1336_133676

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℚ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the polynomial type
def MyPolynomial (α : Type) := ℕ → α

-- Define the condition for the polynomial
def satisfies_condition (f : MyPolynomial ℚ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → f k = 1 / binomial (n + 1) k

-- State the theorem
theorem polynomial_property (n : ℕ) (f : MyPolynomial ℚ) 
  (h : satisfies_condition f n) : 
  f (n + 1) = if n % 2 = 0 then 1 else 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_l1336_133676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_h_domain_range_property_l1336_133617

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x

noncomputable def h (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x + 1) / (f x - 1)

theorem function_properties 
  (k : ℝ) (hk : k ≠ 0) 
  (h_cond : ∀ x, f k (x + 1) * f k x = x^2 + x) :
  (∀ x, f k x = x) ∨ (∀ x, f k x = -x) :=
by sorry

theorem h_domain_range_property :
  ∃ m₁ m₂ : ℝ, 
    (m₁ = -1 ∧ m₂ = 2) ∧
    (∀ x, x ∈ Set.Icc m₁ (m₁ + 1) ↔ h (λ x => x) x ∈ Set.Icc m₁ (m₁ + 1)) ∧
    (∀ x, x ∈ Set.Icc m₂ (m₂ + 1) ↔ h (λ x => x) x ∈ Set.Icc m₂ (m₂ + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_h_domain_range_property_l1336_133617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l1336_133685

theorem graph_translation (φ : ℝ) : 
  (0 < φ) ∧ (φ < π) ∧ 
  (∀ x, Real.sqrt 2 * Real.sin (2*x + π/3) = 2 * Real.sin x * (Real.sin x - Real.cos x) - 1) →
  φ = 13*π/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l1336_133685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_in_interval_l1336_133688

open Real

noncomputable def ω : ℝ := 2

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

noncomputable def g (x : ℝ) : ℝ := f (x - π / 12)

theorem max_value_g_in_interval :
  ∃ (M : ℝ), M = 2 ∧ ∀ x, x ∈ Set.Icc 0 (π / 3) → g x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_in_interval_l1336_133688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1336_133652

/-- A type representing the color of a ball -/
inductive BallColor
| Red
| Black

/-- The contents of the bag -/
def bag : Multiset BallColor := 
  Multiset.replicate 2 BallColor.Red + Multiset.replicate 2 BallColor.Black

/-- A type representing the outcome of drawing two balls -/
structure TwoBallDraw where
  first : BallColor
  second : BallColor

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (draw : TwoBallDraw) : Prop :=
  (draw.first = BallColor.Black ∧ draw.second = BallColor.Red) ∨
  (draw.first = BallColor.Red ∧ draw.second = BallColor.Black)

/-- The event of drawing exactly two black balls -/
def exactlyTwoBlack (draw : TwoBallDraw) : Prop :=
  draw.first = BallColor.Black ∧ draw.second = BallColor.Black

/-- The theorem stating that the events are mutually exclusive but not complementary -/
theorem exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary :
  (∀ draw : TwoBallDraw, ¬(exactlyOneBlack draw ∧ exactlyTwoBlack draw)) ∧
  (∃ draw : TwoBallDraw, ¬exactlyOneBlack draw ∧ ¬exactlyTwoBlack draw) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1336_133652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l1336_133674

/-- If the terminal side of angle α passes through point P(sin(5π/3), cos(5π/3)),
    then sin(π + α) = -1/2 -/
theorem sin_pi_plus_alpha (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = Real.sin (5 * Real.pi / 3) ∧ 
               r * Real.sin α = Real.cos (5 * Real.pi / 3)) → 
  Real.sin (Real.pi + α) = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l1336_133674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_minus_two_zero_on_x_axis_l1336_133627

def point_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

def given_points : List (ℝ × ℝ) := [(0, 2), (-2, -3), (-1, -2), (-2, 0)]

theorem only_minus_two_zero_on_x_axis : 
  ∃! p, p ∈ given_points ∧ point_on_x_axis p ∧ p = (-2, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_minus_two_zero_on_x_axis_l1336_133627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1336_133614

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (t.A > 0 ∧ t.B > 0 ∧ t.C > 0) →
  (t.A + t.B + t.C = Real.pi) →
  (t.a > 0 ∧ t.b > 0 ∧ t.c > 0) →
  (t.a / (Real.sin t.A) = t.b / (Real.sin t.B)) →
  (t.b / (Real.sin t.B) = t.c / (Real.sin t.C)) →
  (Real.cos t.A / (1 + Real.sin t.A) = Real.sin (2 * t.B) / (1 + Real.cos (2 * t.B))) →
  ((t.C = 2 * Real.pi / 3) → (t.B = Real.pi / 6)) ∧
  (∃ (min : ℝ), ∀ (t' : Triangle), 
    (t'.a^2 + t'.b^2) / t'.c^2 ≥ min ∧ 
    min = 4 * Real.sqrt 2 - 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1336_133614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_7_18_l1336_133672

theorem log_7_18 (a b : ℝ) (h1 : Real.log 2 = a * Real.log 10) (h2 : Real.log 3 = b * Real.log 10) :
  Real.log 18 / Real.log 7 = (a + 2*b) / (Real.log 7 / Real.log 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_7_18_l1336_133672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1336_133624

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  (x^(2*y-3)) / ((1/4) + (1/4)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1336_133624
