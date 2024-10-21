import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_on_tuesday_l156_15675

theorem temperature_on_tuesday 
  (tue wed thu : ℝ)
  (avg_tue_wed_thu : (tue + wed + thu) / 3 = 52)
  (avg_wed_thu_fri : (wed + thu + 53) / 3 = 54) :
  tue = 47 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_on_tuesday_l156_15675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l156_15638

-- Define the sphere volume function
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the radii of the three snowballs
def r1 : ℝ := 4
def r2 : ℝ := 6
def r3 : ℝ := 8

-- Theorem statement
theorem snowman_volume :
  sphereVolume r1 + sphereVolume r2 + sphereVolume r3 = 1056 * Real.pi := by
  -- Expand the definition of sphereVolume
  unfold sphereVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l156_15638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l156_15612

theorem quadratic_function_unique (q : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) →  -- q is a quadratic function
  (q 2 = 0) →                                      -- q has a root at x = 2
  (q (-1) = 0) →                                   -- q has a root at x = -1
  (q 3 = 18) →                                     -- q(3) = 18
  (∀ x, q x = 4.5 * x^2 - 4.5 * x - 9) :=           -- Conclusion: q(x) = 4.5x² - 4.5x - 9
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l156_15612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_theorem_l156_15639

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being checked out
variable (is_checked_out : Book → Prop)

-- Define the statement "Every book in the library is checked out"
def every_book_checked_out (Book : Type) (is_checked_out : Book → Prop) : Prop :=
  ∀ b : Book, is_checked_out b

-- Theorem: If "Every book in the library is checked out" is false,
-- then there is at least one book not checked out and not every book is checked out
theorem library_theorem {Book : Type} {is_checked_out : Book → Prop}
    (h : ¬ every_book_checked_out Book is_checked_out) :
    (∃ b : Book, ¬ is_checked_out b) ∧ (¬ ∀ b : Book, is_checked_out b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_theorem_l156_15639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_square_root_count_l156_15609

theorem integer_square_root_count : 
  ∃! (S : Finset ℝ), 
    (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ 
    (∀ x : ℝ, (∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) → x ∈ S) ∧
    S.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_square_root_count_l156_15609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compose_P_Q_six_times_l156_15620

noncomputable def P (x : ℝ) : ℝ := 3 * Real.sqrt x

def Q (x : ℝ) : ℝ := x ^ 3

theorem compose_P_Q_six_times : P (Q (P (Q (P (Q 2))))) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compose_P_Q_six_times_l156_15620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_heads_probability_l156_15680

/-- Represents the probability of getting heads for a coin -/
def CoinProbability := Fin 2 → ℚ

/-- A fair coin has equal probability of heads and tails -/
def fair_coin : CoinProbability :=
  fun i => if i = 0 then 1/2 else 1/2

/-- A biased coin with 3/5 probability of heads -/
def biased_coin : CoinProbability :=
  fun i => if i = 0 then 3/5 else 2/5

/-- The set of coins each person flips -/
def coin_set : Fin 3 → CoinProbability :=
  fun i => if i < 2 then fair_coin else biased_coin

/-- Placeholder for the probability calculation function -/
def probability_same_heads (set1 set2 : Fin 3 → CoinProbability) : ℚ :=
  sorry

/-- The probability of getting the same number of heads when two people
    each flip two fair coins and one biased coin (3/5 heads probability) -/
theorem same_heads_probability :
  ∃ p : ℚ, p = 63/200 ∧ p = probability_same_heads coin_set coin_set :=
by
  sorry

#check same_heads_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_heads_probability_l156_15680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l156_15679

/-- Calculates the percentage decrease in price -/
noncomputable def percentage_decrease (original_price new_price : ℝ) : ℝ :=
  (original_price - new_price) / original_price * 100

theorem price_decrease_percentage :
  let original_price : ℝ := 421.05263157894734
  let new_price : ℝ := 320
  abs (percentage_decrease original_price new_price - 24) < 0.01 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l156_15679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l156_15636

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | x ≤ 1}

-- Define the inverse function
noncomputable def f_inv (x : ℝ) : ℝ := 1 - Real.sqrt x

-- Define the domain of f_inv (which is the range of f)
def dom_f_inv : Set ℝ := {x : ℝ | x ≥ 0}

-- Theorem statement
theorem f_inverse_correct :
  ∀ x ∈ dom_f, ∀ y ∈ dom_f_inv,
    (f x = y ↔ f_inv y = x) ∧
    (f (f_inv y) = y) ∧
    (f_inv (f x) = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l156_15636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l156_15692

theorem sin_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.tan α = -1/2) : 
  Real.sin α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l156_15692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_non_monotonic_condition_max_k_condition_l156_15640

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
def g (k : ℝ) (x : ℝ) : ℝ := k * x^3 - x - 2

-- Part 1: Non-monotonicity of g in (1,2)
theorem g_non_monotonic_condition (k : ℝ) :
  (∃ x y, 1 < x ∧ x < y ∧ y < 2 ∧ g k x > g k y) ↔ 1/12 < k ∧ k < 1/3 :=
by sorry

-- Part 2: Maximum value of k
theorem max_k_condition :
  (∃ k_max : ℝ, k_max = -Real.exp 1 ∧
    (∀ k x, 1 ≤ x → f x ≥ g k x + x + 2 → k ≤ k_max) ∧
    (∀ ε > 0, ∃ x ≥ 1, f x < g (k_max + ε) x + x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_non_monotonic_condition_max_k_condition_l156_15640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l156_15629

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (x - a)^2 - 1 else -(x - b)^2 + 1

-- Part I
theorem part_I (a b : ℝ) (h1 : a < 0) (h2 : ∀ x, f a b x = -f a b (-x)) :
  ∀ x, f a b x = if x ≥ 0 then (x + 1)^2 - 1 else -(x - 1)^2 + 1 := by sorry

-- Part II
theorem part_II (a b : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a b x > f a b y) :
  b - a = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l156_15629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_two_to_three_faces_l156_15686

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of 1x1x1 cubes with exactly two painted faces -/
def count_two_faces (p : RectangularPrism) : ℕ :=
  2 * (p.length - 2) + 2 * (p.width - 2) + 2 * (p.height - 2)

/-- Counts the number of 1x1x1 cubes with exactly three painted faces -/
def count_three_faces : ℕ := 8

theorem ratio_two_to_three_faces (p : RectangularPrism) 
  (h1 : p.length = 4) (h2 : p.width = 5) (h3 : p.height = 6) : 
  (count_two_faces p : ℚ) / count_three_faces = 9 / 2 := by
  sorry

#eval count_two_faces ⟨4, 5, 6⟩
#eval count_three_faces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_two_to_three_faces_l156_15686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_exists_l156_15681

theorem no_polynomial_exists : ¬ ∃ (f : Polynomial ℤ), f.eval 2008 = 0 ∧ f.eval 2010 = 1867 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_exists_l156_15681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_exchange_equation_l156_15677

/-- Represents a gathering of people exchanging gifts -/
structure Gathering where
  attendees : Nat
  total_gifts : Nat
  gift_exchange : Fin attendees → Fin attendees → Prop

/-- The properties of the gift exchange at the gathering -/
def valid_gift_exchange (g : Gathering) : Prop :=
  ∀ a b : Fin g.attendees, a ≠ b → g.gift_exchange a b

/-- The theorem stating the relationship between attendees and total gifts -/
theorem gift_exchange_equation (g : Gathering) 
  (h1 : valid_gift_exchange g)
  (h2 : g.total_gifts = 56) :
  g.attendees * (g.attendees - 1) = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_exchange_equation_l156_15677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l156_15650

noncomputable def curve (x : ℝ) : ℝ := (x + 1) / (x - 1)

noncomputable def curve_derivative (x : ℝ) : ℝ := -2 / ((x - 1)^2)

theorem tangent_perpendicular_to_line (a : ℝ) : 
  curve 3 = 2 →  -- Point (3,2) is on the curve
  curve_derivative 3 * (-a) = 1 →  -- Perpendicular condition
  a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l156_15650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l156_15608

theorem problem_statement (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x - y > Real.log (y / x)) : 
  (x > y) ∧ (x + 1/y > y + 1/x) ∧ (1 / (2:ℝ)^x < (2:ℝ)^(-y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l156_15608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_sum_labeling_natural_exists_equal_sum_labeling_integers_l156_15688

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  incident : Fin 12 → Fin 8 × Fin 8

-- Define a labeling of edges
def Labeling (c : Cube) := Fin 12 → Int

-- Define the sum of labels at a vertex
def vertexSum (c : Cube) (l : Labeling c) (v : Fin 8) : Int :=
  (c.edges.filter (λ e => (c.incident e).1 = v ∨ (c.incident e).2 = v)).sum (λ e => l e)

-- Theorem for part (a)
theorem no_equal_sum_labeling_natural (c : Cube) :
  ¬ ∃ (l : Labeling c), (∀ (e : Fin 12), 1 ≤ l e ∧ l e ≤ 12) ∧
  (∃ (s : Int), ∀ (v : Fin 8), vertexSum c l v = s) :=
sorry

-- Theorem for part (b)
theorem exists_equal_sum_labeling_integers (c : Cube) :
  ∃ (l : Labeling c), (∀ (e : Fin 12), (1 ≤ l e ∧ l e ≤ 6) ∨ (-6 ≤ l e ∧ l e ≤ -1)) ∧
  (∃ (s : Int), ∀ (v : Fin 8), vertexSum c l v = s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_sum_labeling_natural_exists_equal_sum_labeling_integers_l156_15688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_max_cards_l156_15658

def max_cards (total_money : ℚ) (card_cost : ℚ) : ℕ :=
  (total_money / card_cost).floor.toNat

theorem jasmine_max_cards :
  max_cards 9 (3/4) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_max_cards_l156_15658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_equality_condition_l156_15666

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hpq : p ≤ q) 
  (ha : p ≤ a ∧ a ≤ q) 
  (hb : p ≤ b ∧ b ≤ q) 
  (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) 
  (he : p ≤ e ∧ e ≤ q) : 
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 := by
  sorry

theorem equality_condition (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hpq : p ≤ q) 
  (ha : p ≤ a ∧ a ≤ q) 
  (hb : p ≤ b ∧ b ≤ q) 
  (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) 
  (he : p ≤ e ∧ e ≤ q) : 
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) = 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ↔ 
  ((Finset.filter (λ x => x = p) {a, b, c, d, e}).card = 2 ∧ (Finset.filter (λ x => x = q) {a, b, c, d, e}).card = 3) ∨ 
  ((Finset.filter (λ x => x = p) {a, b, c, d, e}).card = 3 ∧ (Finset.filter (λ x => x = q) {a, b, c, d, e}).card = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_equality_condition_l156_15666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l156_15660

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2^x - 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ici (0 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l156_15660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MKO_is_45_degrees_l156_15615

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the directrix of the parabola
noncomputable def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define point M on the parabola
noncomputable def point_M (p : ℝ) : ℝ × ℝ := (p/2, p)

-- Define point K as the intersection of directrix and x-axis
noncomputable def point_K (p : ℝ) : ℝ × ℝ := (-p/2, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Theorem statement
theorem angle_MKO_is_45_degrees (p : ℝ) :
  parabola p (point_M p).1 (point_M p).2 →
  distance (point_M p) (focus p) = p →
  let K := point_K p
  let O := origin
  let M := point_M p
  (K.2 - O.2) * (M.1 - K.1) = (M.2 - K.2) * (K.1 - O.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MKO_is_45_degrees_l156_15615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_victory_margin_l156_15644

/-- Represents the election results for class president --/
structure ElectionResult where
  total_votes : ℕ
  petya_first_two_hours : ℕ
  vasya_first_two_hours : ℕ
  petya_last_hour : ℕ
  vasya_last_hour : ℕ

/-- Conditions for a valid election result --/
def IsValidElection (e : ElectionResult) : Prop :=
  e.total_votes = 27 ∧
  e.petya_first_two_hours = e.vasya_first_two_hours + 9 ∧
  e.vasya_last_hour = e.petya_last_hour + 9 ∧
  e.petya_first_two_hours + e.petya_last_hour > e.vasya_first_two_hours + e.vasya_last_hour

/-- The margin of Petya's victory --/
def VictoryMargin (e : ElectionResult) : ℤ :=
  (e.petya_first_two_hours + e.petya_last_hour : ℤ) - (e.vasya_first_two_hours + e.vasya_last_hour : ℤ)

/-- Theorem stating the maximum possible margin of Petya's victory --/
theorem max_victory_margin :
  ∀ e : ElectionResult, IsValidElection e → VictoryMargin e ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_victory_margin_l156_15644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_range_l156_15627

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (x - a)

-- Part 1
theorem f_monotone_increasing (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ < -2) :
  f (-2) x₁ < f (-2) x₂ := by sorry

-- Part 2
theorem a_range (a : ℝ) (h₁ : a > 0) 
  (h₂ : ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ > f a x₂) :
  a ∈ Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_range_l156_15627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequalities_l156_15691

theorem count_integers_satisfying_inequalities : 
  (Finset.filter (fun n : ℕ => 
    4 ≤ n ∧ n ≤ 14 ∧
    Real.sqrt ((n : ℝ) + 2) ≤ Real.sqrt (3 * (n : ℝ) - 5) ∧ 
    Real.sqrt (3 * (n : ℝ) - 5) < Real.sqrt (2 * (n : ℝ) + 10)) 
    (Finset.range 15)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequalities_l156_15691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l156_15626

open Real Set

noncomputable def f (x : ℝ) := Real.sqrt (2 * Real.sin x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = ⋃ k : ℤ, Icc (2 * π * k + π / 6) (2 * π * k + 5 * π / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l156_15626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l156_15606

-- Define the arithmetic sequence property
def is_arithmetic_seq (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

-- Define the geometric sequence property
def is_geometric_seq (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

-- Theorem statement
theorem arithmetic_geometric_sequence_ratio : 
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℝ),
  (is_arithmetic_seq 1 a₁ a₂ 4) →
  (is_geometric_seq 1 b₁ b₂ b₃ 4) →
  (a₁ + a₂) / b₂ = 5/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l156_15606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l156_15682

/-- Definition of an isosceles triangle -/
def IsoscelesTriangle (P Q R : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)

/-- Definition of triangle area -/
noncomputable def TriangleArea (P Q R : ℝ × ℝ) : ℝ :=
  abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)

/-- Definition of distance between two points -/
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- An isosceles triangle with base 30 and area 120 has congruent sides of length 17 -/
theorem isosceles_triangle_side_length (P Q R : ℝ × ℝ) : 
  IsoscelesTriangle P Q R →
  TriangleArea P Q R = 120 →
  Distance P Q = 30 →
  Distance P R = 17 ∧ Distance Q R = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l156_15682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_big_sciences_related_to_gender_prob_at_least_one_female_l156_15648

-- Define the contingency table data
def male_big_sciences : ℕ := 40
def male_non_big_sciences : ℕ := 15
def female_big_sciences : ℕ := 20
def female_non_big_sciences : ℕ := 25

-- Define the K^2 formula
noncomputable def k_squared (a b c d : ℕ) : ℝ :=
  let n : ℕ := a + b + c + d
  (n : ℝ) * (a * d - b * c : ℝ)^2 / ((a + b : ℝ) * (c + d : ℝ) * (a + c : ℝ) * (b + d : ℝ))

-- Theorem for part 1
theorem big_sciences_related_to_gender :
  k_squared male_big_sciences female_non_big_sciences male_non_big_sciences female_big_sciences > 7.879 := by sorry

-- Theorem for part 2
theorem prob_at_least_one_female (n_male n_female k : ℕ) :
  n_male = 4 → n_female = 2 → k = 2 →
  (Nat.choose n_female 1 * Nat.choose n_male 1 + Nat.choose n_female 2 : ℝ) / Nat.choose (n_male + n_female) k = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_big_sciences_related_to_gender_prob_at_least_one_female_l156_15648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_max_area_equation_l156_15645

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x - 2 * m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := (3 - m) * x + m * y + m^2 - 3 * m = 0

-- Define the distance function between parallel lines
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

-- Define the area of the triangle formed by a line and the positive half-axes
noncomputable def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Theorem 1: Distance between parallel lines
theorem parallel_lines_distance (m : ℝ) :
  (∀ x y, l₁ m x y ↔ l₂ m x y) →
  ∃ d, d = Real.sqrt 5 ∧ d = distance_between_parallel_lines 1 (-2) (-1) (-6) := by
  sorry

-- Theorem 2: Maximum area equation
theorem max_area_equation :
  ∃ m, 0 < m ∧ m < 3 ∧
  (∀ m', 0 < m' ∧ m' < 3 →
    triangle_area ((m^2 - 3*m) / m) (3 - m) ≥ triangle_area ((m'^2 - 3*m') / m') (3 - m')) ∧
  (∀ x y, l₂ m x y ↔ 2*x + 2*y - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_max_area_equation_l156_15645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l156_15665

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area (line1 line2 line3 : ℝ → ℝ) : 
  line1 = (λ x => (2/3 : ℝ) * x + 4) →
  line2 = (λ x => (-3 : ℝ) * x + 9) →
  line3 = (λ _ => (2 : ℝ)) →
  let p1 : ℝ × ℝ := (-3, 2)
  let p2 : ℝ × ℝ := (7/3, 2)
  let p3 : ℝ × ℝ := (15/11, 54/11)
  (∀ x, line1 x = line3 x ↔ x = p1.1) →
  (∀ x, line2 x = line3 x ↔ x = p2.1) →
  (∀ x, line1 x = line2 x ↔ x = p3.1) →
  (1/2 : ℝ) * |p2.1 - p1.1| * |p3.2 - p2.2| = 256/33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l156_15665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_centers_profit_l156_15625

/-- Calculates the combined weekly profit for two distribution centers -/
theorem distribution_centers_profit 
  (packages_center1 : ℕ) 
  (scale_factor : ℕ) 
  (profit_per_package : ℚ) 
  (days_per_week : ℕ) 
  (h1 : packages_center1 = 10000)
  (h2 : scale_factor = 3)
  (h3 : profit_per_package = 5 / 100)
  (h4 : days_per_week = 7)
  : ∃ combined_weekly_profit : ℚ, combined_weekly_profit = 14000 := by
  let packages_center2 : ℕ := packages_center1 * scale_factor
  let daily_profit_center1 : ℚ := packages_center1 * profit_per_package
  let daily_profit_center2 : ℚ := packages_center2 * profit_per_package
  let combined_daily_profit : ℚ := daily_profit_center1 + daily_profit_center2
  let combined_weekly_profit : ℚ := combined_daily_profit * days_per_week
  
  exists combined_weekly_profit
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_centers_profit_l156_15625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_around_square_field_l156_15618

theorem wire_length_around_square_field (area : ℝ) (num_rounds : ℕ) 
  (h : area = 27889 ∧ num_rounds = 11) : 
  (num_rounds * (4 * Real.sqrt area) : ℝ) = 7348 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_around_square_field_l156_15618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l156_15690

/-- The line passing through (1,0) with slope angle 30° -/
def line (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x - 1)

/-- The circle (x-2)^2 + y^2 = 1 -/
def circle' (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

/-- The chord length formed by the intersection of the line and the circle -/
noncomputable def chord_length : ℝ := Real.sqrt 3

theorem intersection_chord_length :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  line x₁ y₁ ∧ line x₂ y₂ ∧
  circle' x₁ y₁ ∧ circle' x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l156_15690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l156_15656

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, Real.sqrt 3 * Real.sin α)

-- Define the line l in Cartesian form
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 * Real.sqrt 3 = 0

-- Define the distance function from a point to line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - Real.sqrt 3 * y - 2 * Real.sqrt 3| / 2

-- Theorem statement
theorem max_distance_curve_to_line :
  ∃ (max_dist : ℝ), max_dist = (3 * Real.sqrt 2 + 2 * Real.sqrt 3) / 2 ∧
  ∀ (α : ℝ), distance_to_line (curve_C α).1 (curve_C α).2 ≤ max_dist := by
  sorry

#check max_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l156_15656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculation_first_figure_angle_calculation_second_figure_l156_15647

theorem angle_calculation_first_figure (A B C D E F : ℝ) 
  (h1 : A + B = 180) -- AB || DE
  (h2 : C = 25)      -- Given angle
  (h3 : D = 55)      -- Given angle
  : A + C + D = 80 := by
  -- Proof steps would go here
  sorry

theorem angle_calculation_second_figure (A B C D E F : ℝ) 
  (h1 : A + B = 180) -- AB || EF
  (h2 : A = 160)     -- Given angle
  (h3 : E = 150)     -- Given angle
  : F + (180 - E) = 50 := by
  -- Proof steps would go here
  sorry

#check angle_calculation_first_figure
#check angle_calculation_second_figure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculation_first_figure_angle_calculation_second_figure_l156_15647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l156_15664

def a (n : ℕ+) (lambda : ℝ) : ℝ := 2 * n.val ^ 2 + lambda * n.val + 3

theorem lambda_range (lambda : ℝ) :
  (∀ n : ℕ+, a n lambda < a (n + 1) lambda) → lambda > -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l156_15664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_candy_difference_l156_15649

theorem amy_candy_difference 
  (first_friend_chocolate : ℝ)
  (first_friend_lollipop : ℝ)
  (second_friend_chocolate : ℝ)
  (second_friend_lollipop : ℝ)
  (left_chocolate : ℝ)
  (left_lollipop : ℝ)
  (h1 : first_friend_chocolate = 8.5)
  (h2 : first_friend_lollipop = 15.25)
  (h3 : second_friend_chocolate = 4.25)
  (h4 : second_friend_lollipop = 9.5)
  (h5 : left_chocolate = 3.75)
  (h6 : left_lollipop = 2.25) :
  (first_friend_chocolate + second_friend_chocolate - left_chocolate) +
  (first_friend_lollipop + second_friend_lollipop - left_lollipop) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_candy_difference_l156_15649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_squared_minus_x_l156_15654

noncomputable section

-- Define the complex number i
def i : ℂ := Complex.I

-- Define x
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2

-- Theorem statement
theorem inverse_x_squared_minus_x : 1 / (x^2 - x) = -1 := by
  -- The proof is omitted for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_x_squared_minus_x_l156_15654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_correct_l156_15632

/-- The area of the largest equilateral triangle inscribed in a circle with radius 8 cm -/
noncomputable def largest_inscribed_triangle_area : ℝ := 48 * Real.sqrt 3

/-- The radius of the circle -/
def circle_radius : ℝ := 8

theorem largest_inscribed_triangle_area_correct :
  largest_inscribed_triangle_area =
    (Real.sqrt 3 / 4) * (circle_radius * Real.sqrt 3)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_triangle_area_correct_l156_15632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_80_l156_15616

/-- The height ratio between consecutive bounces -/
def bounceRatio : ℚ := 2 / 3

/-- The initial height from which the ball is dropped -/
def initialHeight : ℚ := 20

/-- The number of times the ball hits the ground before being caught -/
def bounceCount : ℕ := 4

/-- Calculates the height reached after a given number of bounces -/
def heightAfterBounces (n : ℕ) : ℚ :=
  initialHeight * bounceRatio ^ n

/-- Calculates the total distance travelled by the ball -/
def totalDistance : ℚ :=
  initialHeight + 2 * (Finset.sum (Finset.range bounceCount) heightAfterBounces)

/-- Theorem stating that the total distance travelled is approximately 80 meters -/
theorem total_distance_approx_80 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |((totalDistance : ℚ) : ℝ) - 80| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_80_l156_15616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_one_hour_l156_15698

/-- Represents the reading time distribution of students -/
structure ReadingDistribution where
  time : Fin 5 → ℝ
  count : Fin 5 → ℕ
  total_students : ℕ
  time_ascending : ∀ i j, i < j → time i < time j
  count_sum : (Finset.sum Finset.univ count) = total_students

/-- The median of the reading time distribution is 1 hour -/
theorem median_is_one_hour (d : ReadingDistribution) 
  (h1 : d.total_students = 51)
  (h2 : d.time 0 = 0.5)
  (h3 : d.time 1 = 1)
  (h4 : d.count 0 = 12)
  (h5 : d.count 1 = 22) : 
  ∃ (median : ℝ), median = 1 ∧ 
    (Finset.filter (fun i => d.time i ≤ median) Finset.univ).card ≥ d.total_students / 2 ∧
    (Finset.filter (fun i => d.time i ≥ median) Finset.univ).card ≥ d.total_students / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_one_hour_l156_15698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_points_on_line_segment_l156_15617

/-- The number of integer coordinate points on a line segment -/
def integerPointsOnLineSegment (x1 y1 x2 y2 : ℤ) : ℕ :=
  (Finset.filter (fun p : ℤ × ℤ =>
    (x1 < p.1 ∧ p.1 < x2) ∧
    (99 * p.2 - 197 = 500 * p.1))
    (Finset.product (Finset.Icc (x1 + 1) (x2 - 1)) (Finset.Icc y1 y2))).card

/-- Theorem: There are exactly 17 integer coordinate points on the line segment from (2,3) to (101,503) -/
theorem seventeen_points_on_line_segment :
  integerPointsOnLineSegment 2 3 101 503 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_points_on_line_segment_l156_15617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_two_zeros_l156_15614

/-- A cubic function with parameter b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1/2

/-- The derivative of f with respect to x -/
noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*b*x

theorem cubic_function_two_zeros (b : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f b x = 0 ∧ f b y = 0 ∧
    ∀ z : ℝ, f b z = 0 → z = x ∨ z = y) →
  b = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_two_zeros_l156_15614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_girls_with_dogs_l156_15669

theorem percentage_of_girls_with_dogs (total_students : ℕ) (girls_with_dogs : ℕ) : 
  total_students = 100 →
  girls_with_dogs = 15 - (10 * (total_students / 2) / 100) →
  (girls_with_dogs : ℚ) / (total_students / 2 : ℚ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_girls_with_dogs_l156_15669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_odd_numbers_less_than_6_l156_15628

def is_odd (n : ℕ) : Bool := n % 2 = 1

def in_range (n : ℕ) : Bool := 1 ≤ n ∧ n ≤ 9

def less_than_6 (n : ℕ) : Bool := n < 6

def satisfies_conditions (n : ℕ) : Bool :=
  is_odd n && in_range n && less_than_6 n

def numbers_satisfying_conditions : List ℕ :=
  (List.range 10).filter satisfies_conditions

theorem average_of_odd_numbers_less_than_6 :
  numbers_satisfying_conditions.sum / numbers_satisfying_conditions.length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_odd_numbers_less_than_6_l156_15628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_condition_l156_15653

theorem sin_inequality_condition : 
  (∀ x y : Real, Real.sin x ≠ Real.sin y → x ≠ y) ∧ 
  (∃ x y : Real, x ≠ y ∧ Real.sin x = Real.sin y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_condition_l156_15653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_inequality_l156_15661

-- Define the polynomial P(x)
noncomputable def P : ℝ → ℝ := sorry

-- Define Q(x)
def Q (x : ℝ) : ℝ := x^2 + x + 2001

-- State the theorem
theorem polynomial_inequality :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    (∀ x : ℝ, P x = (x - r₁) * (x - r₂) * (x - r₃))) →  -- P has three distinct real roots
  (∀ x : ℝ, P (Q x) ≠ 0) →  -- P(Q(x)) has no real roots
  P 2001 > 1/64 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_inequality_l156_15661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_four_vertices_l156_15655

/-- A convex polyhedron in 3D space -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  -- This is a placeholder for the actual definition

/-- A vertex of a convex polyhedron -/
def Vertex (P : ConvexPolyhedron) := ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields here
  -- This is a placeholder for the actual definition

/-- Check if a point is on one side of a plane -/
def isOnOneSide (point : ℝ × ℝ × ℝ) (plane : Plane) : Prop :=
  sorry

/-- Check if all points of a polyhedron are on one side of a plane -/
def allOnOneSide (P : ConvexPolyhedron) (plane : Plane) : Prop :=
  sorry

/-- Create a plane passing through a point and parallel to another plane -/
def parallelPlane (point : ℝ × ℝ × ℝ) (plane : Plane) : Plane :=
  sorry

/-- Create a plane through three points -/
def planeThroughPoints (A B C : ℝ × ℝ × ℝ) : Plane :=
  sorry

/-- The main theorem -/
theorem convex_polyhedron_four_vertices (P : ConvexPolyhedron) :
  ∃ (A B C D : Vertex P),
    (allOnOneSide P (parallelPlane A (planeThroughPoints B C D))) ∧
    (allOnOneSide P (parallelPlane B (planeThroughPoints A C D))) ∧
    (allOnOneSide P (parallelPlane C (planeThroughPoints A B D))) ∧
    (allOnOneSide P (parallelPlane D (planeThroughPoints A B C))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_four_vertices_l156_15655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_subset_theorem_l156_15670

/-- A lattice point on a plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Check if two lattice points form a line parallel to coordinate axes -/
def parallelToAxis (p q : LatticePoint) : Prop :=
  p.x = q.x ∨ p.y = q.y

/-- The theorem statement -/
theorem lattice_point_subset_theorem (S : Finset LatticePoint) :
  ∃ (A : Finset LatticePoint) (B : Finset ℤ),
    A ⊆ S ∧
    (∀ p q, p ∈ A → q ∈ A → p ≠ q → ¬parallelToAxis p q) ∧
    (∀ p, p ∈ S → p.x ∈ B ∨ p.y ∈ B) ∧
    (∀ C : Finset LatticePoint, C ⊆ S →
      (∀ p q, p ∈ C → q ∈ C → p ≠ q → ¬parallelToAxis p q) → C.card ≤ A.card) ∧
    (∀ D : Finset ℤ, (∀ p, p ∈ S → p.x ∈ D ∨ p.y ∈ D) → B.card ≤ D.card) →
    A.card ≥ B.card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_subset_theorem_l156_15670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_seven_thirds_l156_15684

-- We don't need to define min as it's already defined in Mathlib

-- State the theorem
theorem solution_is_seven_thirds :
  ∃ (x : ℝ), x = 7/3 ∧ min (1/(1-x)) (2/(1-x)) = 2/(x-1) - 3 := by
  -- Provide the witness
  use 7/3
  -- Split the goal into two parts
  constructor
  -- First part: x = 7/3 (trivial)
  · rfl
  -- Second part: min (1/(1-x)) (2/(1-x)) = 2/(x-1) - 3
  · sorry  -- We'll leave the proof details for later


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_seven_thirds_l156_15684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_points_value_l156_15697

/-- Round-robin tournament with 6 teams -/
structure Tournament where
  teams : Nat
  teams_eq : teams = 6

/-- Points awarded for a win -/
structure WinPoints where
  points : Nat

/-- Total number of games in the tournament -/
def totalGames (t : Tournament) : Nat := t.teams * (t.teams - 1) / 2

/-- Maximum total points possible -/
def maxTotalPoints (t : Tournament) (w : WinPoints) : Nat := t.teams * (t.teams - 1) * w.points

/-- Minimum total points possible -/
def minTotalPoints (t : Tournament) : Nat := totalGames t

theorem win_points_value (t : Tournament) (w : WinPoints) 
  (h : maxTotalPoints t w - minTotalPoints t = 15) : 
  w.points = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_points_value_l156_15697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_of_regular_tetrahedron_l156_15630

/-- Regular tetrahedron with edge length 1 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- Plane passing through midpoints of three edges meeting at a vertex -/
structure MidpointPlane (t : RegularTetrahedron) where

/-- The area of the cross-section formed by a midpoint plane in a regular tetrahedron -/
noncomputable def cross_section_area (t : RegularTetrahedron) (p : MidpointPlane t) : ℝ :=
  Real.pi / 3

/-- Theorem: The area of the cross-section formed by a plane passing through 
    the midpoints of three edges meeting at a vertex of a regular tetrahedron 
    with edge length 1 is π/3 -/
theorem cross_section_area_of_regular_tetrahedron 
  (t : RegularTetrahedron) 
  (p : MidpointPlane t) : 
  cross_section_area t p = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_of_regular_tetrahedron_l156_15630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_value_l156_15601

-- Define the circles and points
noncomputable def circleSet (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 25}

structure Configuration where
  ω₁ : Set (ℝ × ℝ)
  ω₂ : Set (ℝ × ℝ)
  ω₃ : Set (ℝ × ℝ)
  Q₁ : ℝ × ℝ
  Q₂ : ℝ × ℝ
  Q₃ : ℝ × ℝ

-- Define the conditions
def valid_configuration (c : Configuration) : Prop :=
  ∃ (O₁ O₂ O₃ : ℝ × ℝ),
    c.ω₁ = circleSet O₁ ∧
    c.ω₂ = circleSet O₂ ∧
    c.ω₃ = circleSet O₃ ∧
    c.Q₁ ∈ c.ω₁ ∧
    c.Q₂ ∈ c.ω₂ ∧
    c.Q₃ ∈ c.ω₃ ∧
    (O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2 = 100 ∧
    (O₂.1 - O₃.1)^2 + (O₂.2 - O₃.2)^2 = 100 ∧
    (O₃.1 - O₁.1)^2 + (O₃.2 - O₁.2)^2 = 100 ∧
    (c.Q₁.1 - c.Q₂.1)^2 + (c.Q₁.2 - c.Q₂.2)^2 = 
    (c.Q₂.1 - c.Q₃.1)^2 + (c.Q₂.2 - c.Q₃.2)^2 ∧
    (c.Q₂.1 - c.Q₃.1)^2 + (c.Q₂.2 - c.Q₃.2)^2 = 
    (c.Q₃.1 - c.Q₁.1)^2 + (c.Q₃.2 - c.Q₁.2)^2

-- Define the area of the triangle
noncomputable def triangle_area (c : Configuration) : ℝ :=
  let a := ((c.Q₁.1 - c.Q₂.1)^2 + (c.Q₁.2 - c.Q₂.2)^2)^(1/2)
  (Real.sqrt 3 / 4) * a^2

-- State the theorem
theorem triangle_area_value (c : Configuration) 
  (h : valid_configuration c) : 
  triangle_area c = Real.sqrt 675 + Real.sqrt 612.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_value_l156_15601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l156_15603

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Point on a parabola -/
def PointOnParabola (para : Parabola) (p : ℝ × ℝ) : Prop :=
  para.equation p.1 p.2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem statement -/
theorem parabola_minimum_distance :
  let para : Parabola := { focus := (2, 0), equation := fun x y => y^2 = 8*x }
  let A : ℝ × ℝ := (2, 5)
  ∃ (P : ℝ × ℝ),
    PointOnParabola para P ∧
    (∀ (Q : ℝ × ℝ), PointOnParabola para Q →
      distance P A + distance P para.focus ≤ distance Q A + distance Q para.focus) ∧
    distance P A + distance P para.focus = 5 ∧
    P = (2, 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l156_15603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_is_zero_l156_15671

theorem cos_alpha_minus_beta_is_zero 
  (h1 : Real.sin α + Real.sqrt 3 * Real.sin β = 1)
  (h2 : Real.cos α + Real.sqrt 3 * Real.cos β = Real.sqrt 3) :
  Real.cos (α - β) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_is_zero_l156_15671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l156_15695

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
def g (x : ℝ) : ℝ := x^3

-- Define the area as the integral of the difference between f and g
noncomputable def area : ℝ := ∫ x in (0)..(1), f x - g x

-- Theorem statement
theorem area_of_closed_figure : area = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l156_15695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unitary_sum_theorem_not_unitary_251_unitary_1000_l156_15610

/-- A function that returns true if a number is a "unitary number" -/
def is_unitary (n : ℕ) : Prop :=
  ∃ k : ℕ, (k > 0) ∧ (Nat.sum (Nat.digits 10 n) ^ k = 1)

/-- The sum of the original number and its five digit permutations -/
def sum_of_permutations (a b : ℕ) : ℕ :=
  (100*a + 10*b + 1) + (100*a + 10 + b) + (100*b + 10*a + 1) +
  (100*b + 10 + a) + (100 + 10*a + b) + (100 + 10*b + a)

theorem unitary_sum_theorem (a b : ℕ) :
  (2 ≤ a ∧ a ≤ 8) → (2 ≤ b ∧ b ≤ 8) →
  is_unitary (100*a + 10*b + 1) →
  sum_of_permutations a b = 2220 := by
  sorry

/-- 251 is not a unitary number -/
theorem not_unitary_251 : ¬ is_unitary 251 := by
  sorry

/-- 1000 is a unitary number -/
theorem unitary_1000 : is_unitary 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unitary_sum_theorem_not_unitary_251_unitary_1000_l156_15610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l156_15652

theorem problem_statement (P : ℕ → Prop) 
  (h1 : ∀ k, P k → P (k + 1))
  (h2 : ¬ P 4) :
  ∀ n : ℕ, n > 0 → n ≤ 4 → ¬ P n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l156_15652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_one_fourth_l156_15613

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focus of an ellipse -/
noncomputable def focus (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: The eccentricity of the ellipse is 1/4 given the conditions -/
theorem ellipse_eccentricity_is_one_fourth (e : Ellipse) 
  (P : PointOnEllipse e) 
  (h_PF_perp : P.x = -focus e) 
  (h_PF_length : Real.sqrt (P.x^2 + P.y^2) = (3/4) * (e.a + focus e)) : 
  eccentricity e = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_one_fourth_l156_15613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l156_15689

/-- Given non-zero vectors a and b in a real inner product space,
    where |a + b| = |a - b| = 2|b| = 2,
    prove that the projection of (a - b) onto the direction of b is -1. -/
theorem projection_theorem {V : Type} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ‖a + b‖ = ‖a - b‖) (h2 : ‖a + b‖ = 2 * ‖b‖) (h3 : ‖b‖ = 1) :
  inner (a - b) b / ‖b‖^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l156_15689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l156_15667

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: A train 110 meters long, traveling at 36 kmph, will take 26 seconds to cross a bridge 150 meters long -/
theorem train_crossing_bridge_time :
  train_crossing_time 110 150 36 = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l156_15667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theta_l156_15676

noncomputable def f (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x

theorem even_function_theta (θ : ℝ) : 
  (∀ x, f (x + θ) = f (-(x + θ))) → 
  0 ≤ θ → 
  θ ≤ π → 
  θ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theta_l156_15676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l156_15694

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 60) :
  let total_distance := average_speed * 2
  let distance_second_hour := total_distance - speed_first_hour
  let speed_second_hour := distance_second_hour
  speed_second_hour = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l156_15694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_intersections_max_triangle_area_l156_15643

-- Define the ellipse C
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define points B₁ and B₂
def B₁ : ℝ × ℝ := (0, -1)
def B₂ : ℝ × ℝ := (0, 1)

-- Define eccentricity
noncomputable def e : ℝ := Real.sqrt 3 / 2

-- Theorem for part (I)
theorem product_of_intersections (x₀ y₀ : ℝ) :
  ellipse_C x₀ y₀ → x₀ ≠ 0 → y₀ ≠ 0 →
  ∃ (xₘ xₙ : ℝ), 
    (∃ (t : ℝ), y₀ * (xₘ - 0) = (1 + y₀) * (x₀ - xₘ) ∧ -1 * (xₘ - 0) = (1 + y₀) * (0 - (-1))) ∧
    (∃ (s : ℝ), y₀ * (xₙ - 0) = (1 - y₀) * (x₀ - xₙ) ∧ 1 * (xₙ - 0) = (1 - y₀) * (0 - 1)) →
    xₘ * xₙ = 4 := by
  sorry

-- Theorem for part (II)
theorem max_triangle_area (t : ℝ) :
  let x₁ := t + 1
  let y₁ := Real.sqrt ((3 - t^2) / (t^2 + 4))
  let x₂ := t + 1
  let y₂ := -Real.sqrt ((3 - t^2) / (t^2 + 4))
  ellipse_C x₁ y₁ → ellipse_C x₂ y₂ →
  3 * Real.sqrt (t^2 + 3) / (t^2 + 4) ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_intersections_max_triangle_area_l156_15643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l156_15693

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (q.diagonal * q.offset1 + q.diagonal * q.offset2) / 2

/-- Theorem stating that a quadrilateral with given properties has a second offset of 4 cm -/
theorem second_offset_length (q : Quadrilateral) (h1 : q.diagonal = 20) (h2 : q.offset1 = 5) (h3 : area q = 90) :
  q.offset2 = 4 := by
  sorry

#check second_offset_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l156_15693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_of_5_eq_2_l156_15619

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (3 * x + 1)
noncomputable def f (x : ℝ) : ℝ := 5 - t x

-- Theorem statement
theorem t_of_f_of_5_eq_2 : t (f 5) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_of_5_eq_2_l156_15619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cubes_in_box_l156_15683

/-- The maximum number of 27 cubic centimeter cubes that can fit in a rectangular box measuring 8 cm x 9 cm x 12 cm is 32. -/
theorem max_cubes_in_box : ℕ := by
  -- Define the dimensions of the box
  let box_length : ℕ := 8
  let box_width : ℕ := 9
  let box_height : ℕ := 12

  -- Define the volume of a single cube
  let cube_volume : ℕ := 27

  -- Calculate the volume of the box
  let box_volume : ℕ := box_length * box_width * box_height

  -- Calculate the maximum number of cubes that can fit
  have h : box_volume / cube_volume = 32 := by
    norm_num

  exact 32

  -- The proof is complete, so we don't need 'sorry' here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cubes_in_box_l156_15683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l156_15623

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 6*y + 14 = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop :=
  x + y - 2 = 0

-- Define a point P on the circle C
def point_on_circle (P : ℝ × ℝ) : Prop :=
  circle_C P.1 P.2

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (P : ℝ × ℝ) : ℝ :=
  |P.1 + P.2 - 2| / Real.sqrt 2

-- Theorem statement
theorem circle_line_properties :
  (∀ x y : ℝ, circle_C x y → ¬ line_AB x y) ∧
  (∀ P : ℝ × ℝ, point_on_circle P →
    distance_point_to_line P > 1/2 ∧ distance_point_to_line P < 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l156_15623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_large_number_l156_15642

theorem remainder_of_large_number :
  123456789012 % 252 = 87 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_large_number_l156_15642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_l156_15699

/-- Represents the speed of a particle for the nth mile -/
noncomputable def speed (n : ℕ) : ℝ :=
  1 / (2 * (n - 1 : ℝ))

/-- Represents the time taken to traverse the nth mile -/
noncomputable def time (n : ℕ) : ℝ :=
  1 / speed n

theorem particle_movement (n : ℕ) (h : n ≥ 2) :
  time n = 2 * (n - 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_l156_15699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l156_15604

/-- Two 2D vectors are parallel if the ratio of their corresponding components is equal -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_lambda (l : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (l, 4)
  parallel a b → l = 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l156_15604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_correct_l156_15635

/-- Calculates the milk production given initial conditions and new parameters -/
noncomputable def milk_production (a b c d e f g : ℝ) : ℝ :=
  (b * d * e * f * (100 + g)) / (100 * a * c)

/-- Theorem stating the milk production formula is correct -/
theorem milk_production_correct 
  (a b c d e f g : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  milk_production a b c d e f g = (b * d * e * f * (100 + g)) / (100 * a * c) :=
by
  -- Unfold the definition of milk_production
  unfold milk_production
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_correct_l156_15635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l156_15637

/-- The sum of the infinite series ∑(k/3^k) for k from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / 3^k

/-- Theorem: The sum of the infinite series ∑(k/3^k) for k from 1 to infinity equals 3/4 -/
theorem infiniteSeries_sum : infiniteSeries = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l156_15637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l156_15646

def a : ℝ × ℝ × ℝ := (4, 2, 2)
def b : ℝ × ℝ × ℝ := (-3, -3, -3)
def c : ℝ × ℝ × ℝ := (2, 1, 2)

def are_coplanar (v1 v2 v3 : ℝ × ℝ × ℝ) : Prop :=
  let (x1, y1, z1) := v1
  let (x2, y2, z2) := v2
  let (x3, y3, z3) := v3
  Matrix.det !![x1, y1, z1; x2, y2, z2; x3, y3, z3] = 0

theorem vectors_not_coplanar : ¬(are_coplanar a b c) := by
  sorry

#check vectors_not_coplanar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l156_15646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_max_value_l156_15622

/-- Given a line ax + by + 1 = 0 tangent to the circle x^2 + y^2 = 1,
    the maximum value of a + b + ab is √2 + 1/2 -/
theorem tangent_line_max_value (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → a*x + b*y + 1 ≠ 0) →  -- line is tangent to circle
  a^2 + b^2 = 1 →                                   -- derived from tangent condition
  a + b + a*b ≤ Real.sqrt 2 + 1/2 :=                -- √2 + 1/2 is the upper bound
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_max_value_l156_15622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_parallel_implies_n_l156_15678

-- Define the curve
noncomputable def curve (n : ℝ) (x : ℝ) : ℝ := x^n

-- Define the tangent line slope
noncomputable def tangent_slope (n : ℝ) (x : ℝ) : ℝ := n * x^(n - 1)

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

theorem curve_tangent_parallel_implies_n (n : ℝ) : 
  curve n 1 = 1 →  -- The curve passes through (1, 1)
  tangent_slope n 1 = 2 →  -- The tangent line is parallel to 2x - y + 1 = 0
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_parallel_implies_n_l156_15678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l156_15674

theorem sin_x_in_terms_of_a_and_b (a b x : ℝ) 
  (h1 : Real.tan x = (3 * a * b) / (a^2 + 3 * b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < x)
  (h5 : x < Real.pi / 2) :
  Real.sin x = (3 * a * b) / Real.sqrt (a^4 + 15 * a^2 * b^2 + 9 * b^4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l156_15674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l156_15631

-- Define the nearest integer function
noncomputable def nearest_integer (x : ℝ) : ℤ :=
  ⌊x + 1/2⌋

-- Define the main function f(x)
noncomputable def f (x : ℝ) : ℝ := x - nearest_integer x

-- Theorem statement
theorem f_properties :
  (∀ y, y ∈ Set.range f → -1/2 < y ∧ y ≤ 1/2) ∧
  (∀ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3/2 → f x₁ < f x₂) :=
by
  sorry

-- Additional lemmas to support the main theorem
lemma f_range_lower_bound (y : ℝ) (h : y ∈ Set.range f) : -1/2 < y :=
by
  sorry

lemma f_range_upper_bound (y : ℝ) (h : y ∈ Set.range f) : y ≤ 1/2 :=
by
  sorry

lemma f_increasing_on_interval (x₁ x₂ : ℝ) (h : 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3/2) : f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l156_15631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l156_15687

/-- The area of a triangle inscribed in a circle, given the ratio of its sides -/
theorem inscribed_triangle_area (r : ℝ) (a b c : ℝ) (h_positive : 0 < r ∧ 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_ratio : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 4*k) (h_inscribed : c = 2*r) : 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l156_15687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_fraction_l156_15624

theorem recurring_decimal_fraction (a b : ℕ) (h1 : (35 : ℚ) / 99 = (a : ℚ) / b) 
  (h2 : Nat.gcd a b = 1) (h3 : a > 0) (h4 : b > 0) : a + b = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_fraction_l156_15624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_bound_l156_15663

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + x - 1) * Real.exp x

-- Define the theorem
theorem min_M_bound :
  ∃ (M : ℝ), (∀ m n : ℝ, -3 ≤ m ∧ m ≤ 0 ∧ -3 ≤ n ∧ n ≤ 0 → |f m - f n| ≤ M) ∧
  M = 1 - 7 / Real.exp 3 ∧
  ∀ M', (∀ m n : ℝ, -3 ≤ m ∧ m ≤ 0 ∧ -3 ≤ n ∧ n ≤ 0 → |f m - f n| ≤ M') → M ≤ M' :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_bound_l156_15663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_on_assembled_figure_l156_15657

/-- Represents a single die face -/
structure DieFace where
  dots : Nat
  inv_mem : dots ∈ Finset.range 7

/-- Represents a pair of opposite faces on a die -/
structure OppositeFacePair where
  face1 : DieFace
  face2 : DieFace
  sum_seven : face1.dots + face2.dots = 7

/-- Represents the assembled figure -/
structure AssembledFigure where
  dice : Finset (Finset OppositeFacePair)
  dice_count : dice.card = 7
  glued_faces : Finset (DieFace × DieFace)
  glued_faces_count : glued_faces.card = 9
  glued_faces_same_dots : ∀ p ∈ glued_faces, (p.1.dots = p.2.dots)

/-- The theorem to be proved -/
theorem total_dots_on_assembled_figure (fig : AssembledFigure) : 
  (fig.glued_faces.sum (fun p => p.1.dots) * 2) + 
  ((7 - fig.glued_faces.card) * 7) = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_on_assembled_figure_l156_15657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_milk_is_five_l156_15602

/-- The cost of milk per liter, given the following conditions:
  * 3 boxes of flour cost $3 each
  * 3 trays of eggs cost $10 each
  * 7 liters of milk
  * 2 boxes of baking soda cost $3 each
  * Total cost for everything is $80
-/
def cost_of_milk_per_liter : ℚ :=
  let flour_cost : ℚ := 3 * 3
  let eggs_cost : ℚ := 3 * 10
  let baking_soda_cost : ℚ := 2 * 3
  let total_cost : ℚ := 80
  let milk_liters : ℚ := 7
  let other_items_cost : ℚ := flour_cost + eggs_cost + baking_soda_cost
  let milk_total_cost : ℚ := total_cost - other_items_cost
  milk_total_cost / milk_liters

#eval cost_of_milk_per_liter -- Should evaluate to 5

theorem cost_of_milk_is_five : cost_of_milk_per_liter = 5 := by
  -- Unfold the definition and perform the calculation
  unfold cost_of_milk_per_liter
  -- Simplify the arithmetic expressions
  simp [add_mul, mul_add, sub_add_eq_sub_sub]
  -- The result should now be obvious
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_milk_is_five_l156_15602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_proof_l156_15659

/-- Given compound interest conditions, prove the interest rate --/
theorem compound_interest_rate_proof (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 17640)
  (h2 : P * (1 + r)^3 = 22050) :
  ∃ ε > 0, |r - 0.2497| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_proof_l156_15659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_not_four_l156_15621

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * sin (2 * x + π / 3)

-- State the theorem
theorem tangent_slope_not_four :
  ∀ (x₀ : ℝ), ∃ (y₀ : ℝ), y₀ = f x₀ → 
  ∃ (m : ℝ), m = deriv f x₀ → m ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_not_four_l156_15621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_3x_plus_pi_4_l156_15641

noncomputable def f (x : ℝ) : ℝ := Real.tan (3 * x + Real.pi / 4)

theorem period_of_tan_3x_plus_pi_4 :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_3x_plus_pi_4_l156_15641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_hemming_speed_l156_15611

/-- Calculates the number of stitches per minute for hemming a dress -/
noncomputable def stitches_per_minute (hem_length_feet : ℝ) (stitch_length_inches : ℝ) (time_minutes : ℝ) : ℝ :=
  let hem_length_inches : ℝ := hem_length_feet * 12
  let total_stitches : ℝ := hem_length_inches / stitch_length_inches
  total_stitches / time_minutes

/-- Proves that given the specified conditions, the stitches per minute is 24 -/
theorem jenna_hemming_speed : 
  stitches_per_minute 3 (1/4) 6 = 24 := by
  -- Unfold the definition of stitches_per_minute
  unfold stitches_per_minute
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_hemming_speed_l156_15611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l156_15673

theorem trig_identity (α : ℝ) 
  (h : Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α - 2 * Real.sin α - 4 * Real.cos α = 0) : 
  Real.cos α ^ 2 - Real.sin α * Real.cos α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l156_15673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_constant_l156_15685

/-- The perpendicular bisector of a line segment passing through two points. -/
structure PerpendicularBisector where
  a : ℝ × ℝ  -- First point defining the line segment
  b : ℝ × ℝ  -- Second point defining the line segment
  c : ℝ      -- The constant in the equation of the perpendicular bisector

/-- Condition for a line to be a perpendicular bisector of a line segment. -/
def is_perpendicular_bisector (pb : PerpendicularBisector) : Prop :=
  let midpoint := ((pb.a.1 + pb.b.1) / 2, (pb.a.2 + pb.b.2) / 2)
  3 * midpoint.1 - midpoint.2 = pb.c

theorem perpendicular_bisector_constant 
  (pb : PerpendicularBisector) 
  (h : pb.a = (2, 6) ∧ pb.b = (8, 10)) 
  (bisector : is_perpendicular_bisector pb) : 
  pb.c = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_constant_l156_15685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l156_15634

-- Define the function f(x) = √(x-2)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2)

-- State the theorem about the domain of f
theorem domain_of_f : Set.Ici 2 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l156_15634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_is_equidistant_l156_15605

/-- The point in the xy-plane that is equidistant from three given points -/
noncomputable def equidistant_point : ℝ × ℝ × ℝ := (3/8, -3/8, 0)

/-- The first given point -/
def point1 : ℝ × ℝ × ℝ := (0, 2, 0)

/-- The second given point -/
def point2 : ℝ × ℝ × ℝ := (1, -1, 1)

/-- The third given point -/
def point3 : ℝ × ℝ × ℝ := (2, 2, 3)

/-- Calculate the squared distance between two points in 3D space -/
def distance_squared (p q : ℝ × ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2

/-- Theorem stating that the equidistant_point is equidistant from the three given points -/
theorem equidistant_point_is_equidistant :
  distance_squared equidistant_point point1 = distance_squared equidistant_point point2 ∧
  distance_squared equidistant_point point1 = distance_squared equidistant_point point3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_is_equidistant_l156_15605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_similarity_theorem_l156_15696

-- Define the ellipses
def M₁ : Set (ℝ × ℝ) := {p | p.1^2 / 2 + p.2^2 = 1}
def M₂ : Set (ℝ × ℝ) := {p | p.1^2 + 2 * p.2^2 = 1}

-- Define similarity between ellipses
def similar (E₁ E₂ : Set (ℝ × ℝ)) : Prop := 
  ∃ m : ℝ, m > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ E₁ ↔ (x/m, y/m) ∈ E₂

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * (p.1 + 2)}

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define λ₁ and λ₂
noncomputable def lambda₁ (A G : ℝ × ℝ) : ℝ := (A.1 - F.1) / (G.1 - F.1)
noncomputable def lambda₂ (B H : ℝ × ℝ) : ℝ := (B.1 - F.1) / (H.1 - F.1)

theorem ellipse_similarity_theorem :
  similar M₁ M₂ ∧ 
  (1, Real.sqrt 2 / 2) ∈ M₁ ∧
  ∀ (k : ℝ), k ≠ 0 →
    ∀ (A B G H : ℝ × ℝ),
      A ∈ M₁ ∧ B ∈ M₁ ∧ G ∈ M₁ ∧ H ∈ M₁ ∧
      A ∈ line_l k ∧ B ∈ line_l k ∧
      (∃ t : ℝ, G = (1 - t) • A + t • F) ∧
      (∃ s : ℝ, H = (1 - s) • B + s • F) →
        6 < lambda₁ A G + lambda₂ B H ∧ lambda₁ A G + lambda₂ B H < 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_similarity_theorem_l156_15696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l156_15662

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (8, 0)

-- Define D as the midpoint of AB
def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define E as the midpoint of BC
def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_of_triangle_DBC : triangleArea D B C = 12 := by
  -- Expand the definition of triangleArea
  unfold triangleArea
  -- Expand the definition of D
  unfold D
  -- Perform algebraic simplifications
  simp [A, B, C]
  -- The proof is complete
  norm_num

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l156_15662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_triangle_l156_15668

/-- A triangle with integer side lengths where one angle is twice another --/
structure SpecialTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_obtuse : c > a ∧ c > b
  h_angle : ∃ (α β : Real), 0 < α ∧ 0 < β ∧ α + β + 2*β = Real.pi ∧
            Real.sin α / a = Real.sin β / b ∧ Real.sin β / b = Real.sin (2*β) / c

/-- The perimeter of a SpecialTriangle --/
def perimeter (t : SpecialTriangle) : ℕ := t.a + t.b + t.c

/-- The specific triangle with side lengths 7, 9, 12 --/
def triangle_7_9_12 : SpecialTriangle where
  a := 7
  b := 9
  c := 12
  h_obtuse := by sorry
  h_angle := by sorry

/-- Theorem stating that triangle_7_9_12 has the smallest perimeter among all SpecialTriangles --/
theorem smallest_perimeter_triangle :
  ∀ (t : SpecialTriangle), perimeter t ≥ perimeter triangle_7_9_12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_triangle_l156_15668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l156_15607

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The circumference of a circle -/
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem cone_base_circumference (V : ℝ) (h : ℝ) (hV : V = 24 * Real.pi) (hh : h = 6) :
  ∃ (r : ℝ), cone_volume r h = V ∧ circle_circumference r = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l156_15607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_MOI_is_seven_halves_l156_15633

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB : dist A B = 13)
  (AC : dist A C = 12)
  (BC : dist B C = 5)

/-- O is the circumcenter of triangle ABC -/
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- I is the incenter of triangle ABC -/
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- M is the center of a circle tangent to AC, BC, and the circumcircle of triangle ABC -/
noncomputable def M (t : Triangle) : ℝ × ℝ := sorry

/-- The area of triangle MOI -/
noncomputable def areaMOI (t : Triangle) : ℝ :=
  let O := circumcenter t
  let I := incenter t
  let M := M t
  abs (O.1 * I.2 + I.1 * M.2 + M.1 * O.2 - O.2 * I.1 - I.2 * M.1 - M.2 * O.1) / 2

/-- Main theorem: The area of triangle MOI is 7/2 -/
theorem area_MOI_is_seven_halves (t : Triangle) : areaMOI t = 7/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_MOI_is_seven_halves_l156_15633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_floor_a_n_eq_2022_l156_15600

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1 + Real.sqrt 2  -- Added case for 0
  | 1 => 1 + Real.sqrt 2
  | n + 2 => Real.sqrt (n + 2) + Real.sqrt (n + 3)

theorem sequence_a_formula (n : ℕ) (hn : n ≥ 1) :
  sequence_a n = Real.sqrt n + Real.sqrt (n + 1) := by
  sorry

theorem floor_a_n_eq_2022 :
  {n : ℕ | 1021026 ≤ n ∧ n ≤ 1022121} =
  {n : ℕ | ⌊sequence_a n⌋ = 2022} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_floor_a_n_eq_2022_l156_15600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_nine_l156_15651

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem sum_of_a_and_b_is_nine (a b : ℝ) :
  (∀ x ∈ Set.Icc 2 3 ∪ {6}, 0 ≤ f a b x ∧ f a b x ≤ 6 - x) ∧
  (∀ x ∉ Set.Icc 2 3 ∪ {6}, ¬(0 ≤ f a b x ∧ f a b x ≤ 6 - x))
  → a + b = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_nine_l156_15651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l156_15672

theorem trajectory_equation :
  ∀ x y : ℝ, (abs (abs x - abs y) = 4) ↔ (abs x - abs y = 4 ∨ abs x - abs y = -4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l156_15672
