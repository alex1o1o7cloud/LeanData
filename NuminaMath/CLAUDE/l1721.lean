import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_problem_l1721_172164

theorem tangent_line_problem (a : ℝ) :
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → (∀ t : ℝ, t ≠ x → t^3 > m * (t - 1)))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → 
      (∀ t : ℝ, t ≠ x → a * t^2 + (15/4) * t - 9 ≠ m * (t - 1))))) →
  a = -1 ∨ a = -25/64 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1721_172164


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1721_172129

universe u

def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem intersection_complement_equality :
  M ∩ (I \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1721_172129


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l1721_172195

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (a x : ℝ) : ℝ := 2 * |x - a|

-- Question 1
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f x - g 2 x ≤ x - 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

-- Question 2
theorem range_of_a :
  {a : ℝ | ∀ m > 1, ∃ x₀ : ℝ, f x₀ + g a x₀ ≤ (m^2 + m + 4) / (m - 1)} =
  {a : ℝ | -2 - 2 * Real.sqrt 6 ≤ a ∧ a ≤ 2 * Real.sqrt 6 + 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l1721_172195


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_dependency_l1721_172173

/-- Given an arithmetic progression with first term a and common difference d,
    s₁, s₂, and s₄ are the sums of n, 2n, and 4n terms respectively.
    R is defined as s₄ - s₂ - s₁. -/
theorem arithmetic_progression_sum_dependency
  (n : ℕ) (a d : ℝ) 
  (s₁ : ℝ := n * (2 * a + (n - 1) * d) / 2)
  (s₂ : ℝ := 2 * n * (2 * a + (2 * n - 1) * d) / 2)
  (s₄ : ℝ := 4 * n * (2 * a + (4 * n - 1) * d) / 2)
  (R : ℝ := s₄ - s₂ - s₁) :
  R = 6 * d * n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_dependency_l1721_172173


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l1721_172180

theorem cube_sum_divisibility (a b c : ℤ) 
  (h1 : 6 ∣ (a^2 + b^2 + c^2)) 
  (h2 : 3 ∣ (a*b + b*c + c*a)) : 
  6 ∣ (a^3 + b^3 + c^3) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l1721_172180


namespace NUMINAMATH_CALUDE_simultaneous_ring_time_l1721_172134

def library_period : ℕ := 18
def hospital_period : ℕ := 24
def community_center_period : ℕ := 30

def next_simultaneous_ring (t₁ t₂ t₃ : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm t₁ t₂) t₃

theorem simultaneous_ring_time :
  next_simultaneous_ring library_period hospital_period community_center_period = 360 :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_ring_time_l1721_172134


namespace NUMINAMATH_CALUDE_derivative_f_l1721_172111

noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.log (Real.tanh (x/2)) + Real.cosh x - Real.cosh x / (2 * Real.sinh x ^ 2)

theorem derivative_f (x : ℝ) : 
  deriv f x = Real.cosh x ^ 4 / Real.sinh x ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_derivative_f_l1721_172111


namespace NUMINAMATH_CALUDE_paperback_count_l1721_172187

theorem paperback_count (total_books hardbacks selections : ℕ) : 
  total_books = 6 → 
  hardbacks = 4 → 
  selections = 14 →
  (∃ paperbacks : ℕ, 
    paperbacks + hardbacks = total_books ∧
    paperbacks = 2 ↔ 
    (Nat.choose paperbacks 1 * Nat.choose hardbacks 3 +
     Nat.choose paperbacks 2 * Nat.choose hardbacks 2 = selections)) :=
by sorry

end NUMINAMATH_CALUDE_paperback_count_l1721_172187


namespace NUMINAMATH_CALUDE_cubic_function_increasing_l1721_172197

theorem cubic_function_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁^3 < x₂^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_increasing_l1721_172197


namespace NUMINAMATH_CALUDE_total_cookies_l1721_172177

theorem total_cookies (cookies_per_bag : ℕ) (number_of_bags : ℕ) : 
  cookies_per_bag = 41 → number_of_bags = 53 → cookies_per_bag * number_of_bags = 2173 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l1721_172177


namespace NUMINAMATH_CALUDE_fraction_simplification_l1721_172185

theorem fraction_simplification (x y : ℚ) (hx : x = 5) (hy : y = 8) :
  (1 / x - 1 / y) / (1 / x) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1721_172185


namespace NUMINAMATH_CALUDE_student_pairs_l1721_172144

theorem student_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l1721_172144


namespace NUMINAMATH_CALUDE_power_three_mod_thirteen_l1721_172142

theorem power_three_mod_thirteen : 3^39 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_thirteen_l1721_172142


namespace NUMINAMATH_CALUDE_ring_arrangements_l1721_172145

theorem ring_arrangements (n k f : ℕ) (h1 : n = 10) (h2 : k = 7) (h3 : f = 5) :
  let m := (n.choose k) * k.factorial * ((k + f - 1).choose (f - 1))
  (m / 100000000 : ℕ) = 199 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangements_l1721_172145


namespace NUMINAMATH_CALUDE_tan_equation_solution_l1721_172124

theorem tan_equation_solution (x : ℝ) : 
  x = 30 * Real.pi / 180 → 
  Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l1721_172124


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1721_172175

/-- Given an arithmetic sequence with first four terms a, x, b, 2x, prove that a/b = 1/3 -/
theorem arithmetic_sequence_ratio (a x b : ℝ) :
  (x - a = b - x) ∧ (b - x = 2 * x - b) → a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1721_172175


namespace NUMINAMATH_CALUDE_s_one_eq_one_l1721_172148

/-- s(n) is a function that returns the n-digit number formed by attaching
    the first n perfect squares in order. -/
def s (n : ℕ) : ℕ := sorry

/-- Theorem: s(1) equals 1 -/
theorem s_one_eq_one : s 1 = 1 := by sorry

end NUMINAMATH_CALUDE_s_one_eq_one_l1721_172148


namespace NUMINAMATH_CALUDE_cheryl_mm_theorem_l1721_172105

def cheryl_mm_problem (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) (remaining : ℕ) : Prop :=
  initial - eaten_lunch - eaten_dinner - remaining = 18

theorem cheryl_mm_theorem :
  cheryl_mm_problem 40 7 5 10 := by sorry

end NUMINAMATH_CALUDE_cheryl_mm_theorem_l1721_172105


namespace NUMINAMATH_CALUDE_A_divisibility_l1721_172157

/-- Definition of A_l for a prime p > 3 -/
def A (p : ℕ) (l : ℕ) : ℕ :=
  sorry

/-- Theorem statement -/
theorem A_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  (∀ l, 1 ≤ l ∧ l ≤ p - 2 → p ∣ A p l) ∧
  (∀ l, 1 < l ∧ l < p ∧ Odd l → p^2 ∣ A p l) :=
by sorry

end NUMINAMATH_CALUDE_A_divisibility_l1721_172157


namespace NUMINAMATH_CALUDE_sine_special_angle_l1721_172121

theorem sine_special_angle (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (-α - π) = Real.sqrt 5 / 5) : 
  Real.sin (α - 3 * π / 2) = -(2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_special_angle_l1721_172121


namespace NUMINAMATH_CALUDE_equation_solutions_l1721_172135

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧ 
    x₁^2 - 4*x₁ + 1 = 0 ∧ x₂^2 - 4*x₂ + 1 = 0) ∧
  (∃ (x₃ x₄ : ℝ), x₃ = 2/5 ∧ x₄ = -5/3 ∧ 
    5*x₃ - 2 = (2 - 5*x₃)*(3*x₃ + 4) ∧ 5*x₄ - 2 = (2 - 5*x₄)*(3*x₄ + 4)) :=
by sorry


end NUMINAMATH_CALUDE_equation_solutions_l1721_172135


namespace NUMINAMATH_CALUDE_find_g_x_l1721_172143

/-- Given that 4x^4 - 6x^2 + 2 + g(x) = 7x^3 - 3x^2 + 4x - 1 for all x,
    prove that g(x) = -4x^4 + 7x^3 + 3x^2 + 4x - 3 -/
theorem find_g_x (g : ℝ → ℝ) :
  (∀ x, 4 * x^4 - 6 * x^2 + 2 + g x = 7 * x^3 - 3 * x^2 + 4 * x - 1) →
  (∀ x, g x = -4 * x^4 + 7 * x^3 + 3 * x^2 + 4 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_find_g_x_l1721_172143


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1721_172174

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (matches_with_known_average : ℕ) 
  (known_average : ℝ) 
  (total_average : ℝ) 
  (h1 : total_matches = 25)
  (h2 : matches_with_known_average = 15)
  (h3 : known_average = 70)
  (h4 : total_average = 66) :
  let remaining_matches := total_matches - matches_with_known_average
  (total_matches * total_average - matches_with_known_average * known_average) / remaining_matches = 60 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1721_172174


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1721_172114

/-- Represents a point in 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semimajor : ℚ
  semiminor : ℚ

/-- Check if a point lies on the ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.semimajor^2 + (p.y - e.center.y)^2 / e.semiminor^2 = 1

/-- The six given points -/
def points : List Point := [
  ⟨-5/2, 2⟩, ⟨0, 0⟩, ⟨0, 3⟩, ⟨4, 0⟩, ⟨4, 3⟩, ⟨2, 4⟩
]

/-- The ellipse passing through the points -/
def ellipse : Ellipse := ⟨⟨2, 3/2⟩, 2, 5/2⟩

theorem ellipse_minor_axis_length :
  (∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ (∃ (m b : ℚ), p1.y = m * p1.x + b ∧ p2.y = m * p2.x + b ∧ p3.y = m * p3.x + b)) →
  (∀ p : Point, p ∈ points → pointOnEllipse p ellipse) →
  ellipse.semiminor * 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1721_172114


namespace NUMINAMATH_CALUDE_triangle_properties_l1721_172149

/-- Theorem about properties of an acute triangle ABC --/
theorem triangle_properties 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle opposite to A, B, C respectively
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) -- Triangle is acute
  (h_sine : Real.sqrt 3 * a = 2 * c * Real.sin A) -- Given condition
  (h_side : a = 2) -- Given side length
  (h_area : (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2) -- Given area
  : C = π/3 ∧ c = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1721_172149


namespace NUMINAMATH_CALUDE_min_k_for_inequality_l1721_172101

theorem min_k_for_inequality (x y : ℝ) : 
  x * (x - 1) ≤ y * (1 - y) → 
  (∃ k : ℝ, (∀ x y : ℝ, x * (x - 1) ≤ y * (1 - y) → x^2 + y^2 ≤ k) ∧ 
   (∀ k' : ℝ, k' < k → ∃ x y : ℝ, x * (x - 1) ≤ y * (1 - y) ∧ x^2 + y^2 > k')) ∧
  (∀ k : ℝ, (∀ x y : ℝ, x * (x - 1) ≤ y * (1 - y) → x^2 + y^2 ≤ k) → k ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l1721_172101


namespace NUMINAMATH_CALUDE_cody_tickets_l1721_172125

/-- Calculates the final number of tickets Cody has after winning, spending, and winning again. -/
def final_tickets (initial : ℕ) (spent : ℕ) (won_later : ℕ) : ℕ :=
  initial - spent + won_later

/-- Theorem stating that Cody ends up with 30 tickets given the problem conditions. -/
theorem cody_tickets : final_tickets 49 25 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cody_tickets_l1721_172125


namespace NUMINAMATH_CALUDE_proposition_logic_l1721_172171

theorem proposition_logic (p q : Prop) (hp : p = (2 + 2 = 5)) (hq : q = (3 > 2)) :
  (p ∨ q) ∧ ¬(¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_logic_l1721_172171


namespace NUMINAMATH_CALUDE_wilsborough_savings_l1721_172136

/-- Mrs. Wilsborough's concert ticket purchase problem -/
theorem wilsborough_savings (vip_price regular_price : ℕ) 
  (vip_count regular_count leftover : ℕ) :
  vip_price = 100 →
  regular_price = 50 →
  vip_count = 2 →
  regular_count = 3 →
  leftover = 150 →
  vip_count * vip_price + regular_count * regular_price + leftover = 500 :=
by sorry

end NUMINAMATH_CALUDE_wilsborough_savings_l1721_172136


namespace NUMINAMATH_CALUDE_final_amounts_l1721_172161

/-- Represents a person with their current amount of money -/
structure Person where
  name : String
  amount : ℚ

/-- Represents the state of all persons involved in the transactions -/
structure State where
  michael : Person
  thomas : Person
  emily : Person

/-- Performs the series of transactions described in the problem -/
def performTransactions (initial : State) : State :=
  let s1 := { initial with
    michael := { initial.michael with amount := initial.michael.amount * (1 - 0.3) },
    thomas := { initial.thomas with amount := initial.thomas.amount + initial.michael.amount * 0.3 }
  }
  let s2 := { s1 with
    thomas := { s1.thomas with amount := s1.thomas.amount * (1 - 0.25) },
    emily := { s1.emily with amount := s1.emily.amount + s1.thomas.amount * 0.25 }
  }
  let s3 := { s2 with
    emily := { s2.emily with amount := (s2.emily.amount - 10) / 2 },
    michael := { s2.michael with amount := s2.michael.amount + (s2.emily.amount - 10) / 2 }
  }
  s3

/-- The main theorem stating the final amounts after transactions -/
theorem final_amounts (initial : State)
  (h_michael : initial.michael.amount = 42)
  (h_thomas : initial.thomas.amount = 17)
  (h_emily : initial.emily.amount = 30) :
  let final := performTransactions initial
  final.michael.amount = 43.1 ∧
  final.thomas.amount = 22.2 ∧
  final.emily.amount = 13.7 := by
  sorry


end NUMINAMATH_CALUDE_final_amounts_l1721_172161


namespace NUMINAMATH_CALUDE_jack_john_vote_difference_l1721_172167

/-- Calculates the number of votes Jack received more than John in an election with given conditions. -/
theorem jack_john_vote_difference :
  let total_votes : ℕ := 1150
  let john_votes : ℕ := 150
  let remaining_votes : ℕ := total_votes - john_votes
  let james_votes : ℕ := (7 * remaining_votes) / 10
  let jacob_votes : ℕ := (3 * (john_votes + james_votes)) / 10
  let joey_votes : ℕ := ((125 * jacob_votes) + 50) / 100
  let jack_votes : ℕ := (95 * joey_votes) / 100
  jack_votes - john_votes = 153 := by sorry

end NUMINAMATH_CALUDE_jack_john_vote_difference_l1721_172167


namespace NUMINAMATH_CALUDE_subset_intersection_condition_l1721_172132

theorem subset_intersection_condition (M N : Set α) (h_nonempty : M.Nonempty) (h_subset : M ⊆ N) :
  (∀ a, a ∈ M ∩ N → (a ∈ M ∨ a ∈ N)) ∧
  ¬(∀ a, (a ∈ M ∨ a ∈ N) → a ∈ M ∩ N) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_condition_l1721_172132


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1721_172162

theorem ratio_w_to_y (w x y z : ℝ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1721_172162


namespace NUMINAMATH_CALUDE_polygon_sides_l1721_172115

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1721_172115


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_property_l1721_172126

theorem quadratic_equation_roots_property : ∃ (p q : ℝ),
  p + q = 7 ∧
  |p - q| = 9 ∧
  ∀ x, x^2 - 7*x - 8 = 0 ↔ (x = p ∨ x = q) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_property_l1721_172126


namespace NUMINAMATH_CALUDE_emily_dresses_l1721_172191

theorem emily_dresses (melissa : ℕ) (debora : ℕ) (emily : ℕ) : 
  debora = melissa + 12 →
  melissa = emily / 2 →
  melissa + debora + emily = 44 →
  emily = 16 := by sorry

end NUMINAMATH_CALUDE_emily_dresses_l1721_172191


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l1721_172102

/-- 
Given a quadratic equation (a-1)x^2 - 4x - 1 = 0, where 'a' is a parameter,
this theorem states the conditions on 'a' for the equation to have two real roots.
-/
theorem quadratic_two_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 4*x - 1 = 0 ∧ (a - 1) * y^2 - 4*y - 1 = 0) ↔ 
  (a ≥ -3 ∧ a ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l1721_172102


namespace NUMINAMATH_CALUDE_brianna_book_purchase_l1721_172122

theorem brianna_book_purchase (total_money : ℚ) (total_books : ℚ) :
  total_money > 0 ∧ total_books > 0 →
  (1 / 4 : ℚ) * total_money = (1 / 2 : ℚ) * total_books →
  total_money - 2 * ((1 / 4 : ℚ) * total_money) = (1 / 2 : ℚ) * total_money :=
by sorry

end NUMINAMATH_CALUDE_brianna_book_purchase_l1721_172122


namespace NUMINAMATH_CALUDE_park_diameter_l1721_172151

/-- Given a circular park with concentric rings, calculate the diameter of the outer boundary. -/
theorem park_diameter (statue_width garden_width path_width fountain_diameter : ℝ) : 
  statue_width = 2 ∧ 
  garden_width = 10 ∧ 
  path_width = 8 ∧ 
  fountain_diameter = 12 → 
  2 * (fountain_diameter / 2 + statue_width + garden_width + path_width) = 52 := by
sorry

end NUMINAMATH_CALUDE_park_diameter_l1721_172151


namespace NUMINAMATH_CALUDE_four_propositions_true_l1721_172154

theorem four_propositions_true (x y : ℝ) : 
  (((x = 0 ∧ y = 0) → (x^2 + y^2 ≠ 0)) ∧                   -- Original
   ((x^2 + y^2 ≠ 0) → (x = 0 ∧ y = 0)) ∧                   -- Converse
   (¬(x = 0 ∧ y = 0) → ¬(x^2 + y^2 ≠ 0)) ∧                 -- Inverse
   (¬(x^2 + y^2 ≠ 0) → ¬(x = 0 ∧ y = 0)))                  -- Contrapositive
  := by sorry

end NUMINAMATH_CALUDE_four_propositions_true_l1721_172154


namespace NUMINAMATH_CALUDE_sqrt_7_simplest_l1721_172140

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ 
  (∀ (z : ℕ), z > 1 → ¬(∃ (w : ℝ), y = z * w ^ 2)) ∧
  y ≠ 1

theorem sqrt_7_simplest : 
  is_simplest_quadratic_radical (Real.sqrt 7) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt 4)) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt (1/4))) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt 27)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_7_simplest_l1721_172140


namespace NUMINAMATH_CALUDE_peaches_at_stand_l1721_172190

/-- The total number of peaches at the stand after picking more is equal to the sum of the initial number of peaches and the number of peaches picked. -/
theorem peaches_at_stand (initial_peaches picked_peaches : ℕ) :
  initial_peaches + picked_peaches = initial_peaches + picked_peaches :=
by sorry

end NUMINAMATH_CALUDE_peaches_at_stand_l1721_172190


namespace NUMINAMATH_CALUDE_f_deriv_negative_one_eq_negative_two_l1721_172104

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- State the theorem
theorem f_deriv_negative_one_eq_negative_two 
  (a b c : ℝ) (h : f_deriv a b 1 = 2) : f_deriv a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_deriv_negative_one_eq_negative_two_l1721_172104


namespace NUMINAMATH_CALUDE_find_d_l1721_172106

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x - 3

-- State the theorem
theorem find_d (c : ℝ) :
  (∃ d : ℝ, ∀ x : ℝ, f c (g c x) = 15 * x + d) →
  (∃ d : ℝ, ∀ x : ℝ, f c (g c x) = 15 * x + d ∧ d = -12) :=
by sorry

end NUMINAMATH_CALUDE_find_d_l1721_172106


namespace NUMINAMATH_CALUDE_factors_of_1320_l1721_172179

/-- The number of distinct positive factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- 1320 is our number of interest -/
def our_number : ℕ := 1320

/-- Theorem stating that 1320 has 32 distinct positive factors -/
theorem factors_of_1320 : num_factors our_number = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l1721_172179


namespace NUMINAMATH_CALUDE_guys_with_bullets_l1721_172131

theorem guys_with_bullets (n : ℕ) (h : n > 0) : 
  (∀ (guy : Fin n), 25 - 4 = (n * 25 - n * 4) / n) → n ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_guys_with_bullets_l1721_172131


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1721_172112

def is_valid_external_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  a^2 + c^2 > b^2 ∧
  b^2 + c^2 > a^2

theorem invalid_external_diagonals :
  ¬ (is_valid_external_diagonals 5 6 9) :=
by sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1721_172112


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1721_172193

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x > 6} = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1721_172193


namespace NUMINAMATH_CALUDE_vessel_capacity_proof_l1721_172127

/-- Proves that the capacity of the first vessel is 2 liters given the problem conditions -/
theorem vessel_capacity_proof (
  first_vessel_alcohol_percentage : ℝ)
  (second_vessel_capacity : ℝ)
  (second_vessel_alcohol_percentage : ℝ)
  (total_liquid_poured : ℝ)
  (new_vessel_capacity : ℝ)
  (new_mixture_alcohol_percentage : ℝ)
  (h1 : first_vessel_alcohol_percentage = 0.20)
  (h2 : second_vessel_capacity = 6)
  (h3 : second_vessel_alcohol_percentage = 0.55)
  (h4 : total_liquid_poured = 8)
  (h5 : new_vessel_capacity = 10)
  (h6 : new_mixture_alcohol_percentage = 0.37)
  : ∃ (first_vessel_capacity : ℝ),
    first_vessel_capacity = 2 ∧
    first_vessel_capacity * first_vessel_alcohol_percentage +
    second_vessel_capacity * second_vessel_alcohol_percentage =
    new_vessel_capacity * new_mixture_alcohol_percentage :=
by sorry

end NUMINAMATH_CALUDE_vessel_capacity_proof_l1721_172127


namespace NUMINAMATH_CALUDE_age_of_twentieth_student_l1721_172194

theorem age_of_twentieth_student (total_students : Nat) (total_avg_age : Nat)
  (group1_count : Nat) (group1_avg_age : Nat)
  (group2_count : Nat) (group2_avg_age : Nat)
  (group3_count : Nat) (group3_avg_age : Nat) :
  total_students = 20 →
  total_avg_age = 18 →
  group1_count = 6 →
  group1_avg_age = 16 →
  group2_count = 8 →
  group2_avg_age = 17 →
  group3_count = 5 →
  group3_avg_age = 21 →
  (total_students * total_avg_age) - 
  (group1_count * group1_avg_age + group2_count * group2_avg_age + group3_count * group3_avg_age) = 23 := by
  sorry

end NUMINAMATH_CALUDE_age_of_twentieth_student_l1721_172194


namespace NUMINAMATH_CALUDE_range_of_a_l1721_172165

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem range_of_a :
  (∀ x, ¬(p x) → ¬(q x a)) ∧
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1721_172165


namespace NUMINAMATH_CALUDE_perpendicular_chords_diameter_l1721_172146

theorem perpendicular_chords_diameter (r : ℝ) (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  a + b = 7 →
  c + d = 8 →
  (a * b = r^2) ∧ (c * d = r^2) →
  2 * r = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_chords_diameter_l1721_172146


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l1721_172137

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l1721_172137


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1721_172172

theorem fraction_to_decimal : (45 : ℚ) / (2^3 * 5^4) = (9 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1721_172172


namespace NUMINAMATH_CALUDE_min_value_inequality_l1721_172176

def f (k : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + k|

theorem min_value_inequality (k : ℝ) (a b c : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x, f k x ≥ 3)
  (h3 : ∃ x, f k x = 3)
  (h4 : a + b + c = k) :
  a^2 + b^2 + c^2 ≥ 4/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1721_172176


namespace NUMINAMATH_CALUDE_two_transformations_preserve_pattern_l1721_172181

/-- Represents the pattern of squares on the infinite line -/
structure SquarePattern where
  s : ℝ  -- side length of each square
  ℓ : Line2  -- the infinite line

/-- Enumeration of the four transformations -/
inductive Transformation
  | rotation180 : Point → Transformation
  | translation4s : Transformation
  | reflectionAcrossL : Transformation
  | reflectionPerpendicular : Point → Transformation

/-- Predicate to check if a transformation maps the pattern onto itself -/
def mapsOntoItself (t : Transformation) (p : SquarePattern) : Prop :=
  sorry

theorem two_transformations_preserve_pattern (p : SquarePattern) :
  ∃! (ts : Finset Transformation), ts.card = 2 ∧
    ∀ t ∈ ts, mapsOntoItself t p ∧
    ∀ t, mapsOntoItself t p → t ∈ ts :=
  sorry

end NUMINAMATH_CALUDE_two_transformations_preserve_pattern_l1721_172181


namespace NUMINAMATH_CALUDE_number_of_girls_l1721_172113

theorem number_of_girls (total_pupils : ℕ) (boys : ℕ) (teachers : ℕ) 
  (h1 : total_pupils = 626)
  (h2 : boys = 318)
  (h3 : teachers = 36) :
  total_pupils - boys - teachers = 272 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l1721_172113


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1721_172109

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  2 * ((1 / 2) * a 3) = 3 * a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticCondition a →
  (a 20 + a 19) / (a 18 + a 17) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1721_172109


namespace NUMINAMATH_CALUDE_set_operations_l1721_172169

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≤ 8 - 2 * x}

-- State the theorem
theorem set_operations :
  (B = {x : ℝ | x ≤ 3}) ∧
  (A ∪ B = {x : ℝ | x < 4}) ∧
  ((Aᶜ) ∩ B = {x : ℝ | x < -1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1721_172169


namespace NUMINAMATH_CALUDE_power_of_one_third_l1721_172128

theorem power_of_one_third (a b : ℕ) : 
  (2^a = 8 ∧ 5^b = 25) → (1/3 : ℚ)^(b - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_third_l1721_172128


namespace NUMINAMATH_CALUDE_base_five_representation_of_156_l1721_172182

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBaseFive (n / 5)

/-- Checks if a list of digits represents a valid base 5 number --/
def isValidBaseFive (digits : List ℕ) : Prop :=
  digits.all (· < 5)

theorem base_five_representation_of_156 :
  let base5Repr := toBaseFive 156
  isValidBaseFive base5Repr ∧ base5Repr = [1, 1, 1, 1] := by
  sorry

#eval toBaseFive 156  -- Should output [1, 1, 1, 1]

end NUMINAMATH_CALUDE_base_five_representation_of_156_l1721_172182


namespace NUMINAMATH_CALUDE_pig_bacon_profit_l1721_172139

def average_pig_bacon : ℝ := 20
def average_type_a_bacon : ℝ := 12
def average_type_b_bacon : ℝ := 8
def type_a_price : ℝ := 6
def type_b_price : ℝ := 4
def this_pig_size_ratio : ℝ := 0.5
def this_pig_type_a_ratio : ℝ := 0.75
def this_pig_type_b_ratio : ℝ := 0.25
def type_a_cost : ℝ := 1.5
def type_b_cost : ℝ := 0.8

theorem pig_bacon_profit : 
  let this_pig_bacon := average_pig_bacon * this_pig_size_ratio
  let this_pig_type_a := this_pig_bacon * this_pig_type_a_ratio
  let this_pig_type_b := this_pig_bacon * this_pig_type_b_ratio
  let revenue := this_pig_type_a * type_a_price + this_pig_type_b * type_b_price
  let cost := this_pig_type_a * type_a_cost + this_pig_type_b * type_b_cost
  revenue - cost = 41.75 := by
sorry

end NUMINAMATH_CALUDE_pig_bacon_profit_l1721_172139


namespace NUMINAMATH_CALUDE_triangle_hypotenuse_length_l1721_172155

-- Define the triangle and points
def Triangle (P Q R : ℝ × ℝ) : Prop := sorry

def RightTriangle (P Q R : ℝ × ℝ) : Prop := 
  Triangle P Q R ∧ sorry -- Add condition for right angle

def PointOnLine (P Q M : ℝ × ℝ) : Prop := sorry

-- Define the ratio condition
def RatioCondition (P M Q : ℝ × ℝ) : Prop := 
  ∃ (k : ℝ), k = 1/3 ∧ sorry -- Add condition for PM:MQ = 1:3

-- Define the distance function
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_hypotenuse_length 
  (P Q R M N : ℝ × ℝ) 
  (h1 : RightTriangle P Q R) 
  (h2 : PointOnLine P Q M) 
  (h3 : PointOnLine P R N) 
  (h4 : RatioCondition P M Q) 
  (h5 : RatioCondition P N R) 
  (h6 : distance Q N = 20) 
  (h7 : distance M R = 36) : 
  distance Q R = 2 * Real.sqrt 399 := by
  sorry

end NUMINAMATH_CALUDE_triangle_hypotenuse_length_l1721_172155


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1721_172188

theorem sin_cos_identity : Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (200 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1721_172188


namespace NUMINAMATH_CALUDE_system_solution_l1721_172159

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -8) ∧
    (5 * x + 9 * y = -18) ∧
    (x = -14/3) ∧
    (y = -32/9) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1721_172159


namespace NUMINAMATH_CALUDE_sum_difference_problem_l1721_172183

theorem sum_difference_problem (x y : ℤ) : x + y = 45 → x = 25 → x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_problem_l1721_172183


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1721_172158

theorem magnitude_of_z (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1721_172158


namespace NUMINAMATH_CALUDE_count_squarish_numbers_l1721_172160

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_two_digit_perfect_square (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_perfect_square n

def is_squarish (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  is_perfect_square n ∧
  n % 16 = 0 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  is_two_digit_perfect_square (n / 10000) ∧
  is_two_digit_perfect_square ((n / 100) % 100) ∧
  is_two_digit_perfect_square (n % 100)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_squarish_numbers_l1721_172160


namespace NUMINAMATH_CALUDE_investment_principal_l1721_172110

/-- Proves that an investment with a 9% simple annual interest rate yielding $231 monthly interest has a principal of $30,800 --/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 231 →
  annual_rate = 0.09 →
  (monthly_interest / (annual_rate / 12)) = 30800 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_l1721_172110


namespace NUMINAMATH_CALUDE_a5_greater_than_b5_l1721_172118

-- Define the geometric sequence a_n
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Define the arithmetic sequence b_n
def arithmetic_sequence (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  b₁ + (n - 1) * d

theorem a5_greater_than_b5 
  (a₁ b₁ q d : ℝ)
  (h1 : a₁ = b₁)
  (h2 : a₁ > 0)
  (h3 : geometric_sequence a₁ q 3 = arithmetic_sequence b₁ d 3)
  (h4 : a₁ ≠ geometric_sequence a₁ q 3) :
  geometric_sequence a₁ q 5 > arithmetic_sequence b₁ d 5 := by
  sorry

end NUMINAMATH_CALUDE_a5_greater_than_b5_l1721_172118


namespace NUMINAMATH_CALUDE_modulo_congruence_l1721_172108

theorem modulo_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 100000 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_l1721_172108


namespace NUMINAMATH_CALUDE_smallest_r_is_pi_over_two_l1721_172141

theorem smallest_r_is_pi_over_two :
  ∃ (r : ℝ) (f g : ℝ → ℝ), r > 0 ∧
    Differentiable ℝ f ∧ Differentiable ℝ g ∧
    f 0 > 0 ∧
    g 0 = 0 ∧
    (∀ x, |deriv f x| ≤ |g x|) ∧
    (∀ x, |deriv g x| ≤ |f x|) ∧
    f r = 0 ∧
    (∀ r' > 0, (∃ f' g' : ℝ → ℝ,
      Differentiable ℝ f' ∧ Differentiable ℝ g' ∧
      f' 0 > 0 ∧
      g' 0 = 0 ∧
      (∀ x, |deriv f' x| ≤ |g' x|) ∧
      (∀ x, |deriv g' x| ≤ |f' x|) ∧
      f' r' = 0) → r' ≥ r) ∧
    r = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_r_is_pi_over_two_l1721_172141


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1721_172107

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a : ℚ) (b : ℚ) :
  a = 3/4 → b = 1 → arithmetic_sequence a (b - a) 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1721_172107


namespace NUMINAMATH_CALUDE_unique_two_digit_reverse_pair_l1721_172184

theorem unique_two_digit_reverse_pair (z : ℕ) (h : z ≥ 3) :
  ∃! (A B : ℕ),
    (A < z^2 ∧ A ≥ z) ∧
    (B < z^2 ∧ B ≥ z) ∧
    (∃ (p q : ℕ), A = p * z + q ∧ B = q * z + p) ∧
    (∀ x : ℝ, (x^2 - A*x + B = 0) → (∃! r : ℝ, x = r)) ∧
    A = (z - 1)^2 ∧
    B = 2*(z - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_reverse_pair_l1721_172184


namespace NUMINAMATH_CALUDE_two_element_subsets_of_three_element_set_l1721_172163

theorem two_element_subsets_of_three_element_set :
  let S : Finset Int := {-1, 0, 2}
  (Finset.filter (fun M => M.card = 2) (S.powerset)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_two_element_subsets_of_three_element_set_l1721_172163


namespace NUMINAMATH_CALUDE_tom_purchases_amount_l1721_172123

/-- Calculates the amount available for purchases given hourly rate, work hours, and savings rate. -/
def amountAvailableForPurchases (hourlyRate : ℚ) (workHours : ℕ) (savingsRate : ℚ) : ℚ :=
  let totalEarnings := hourlyRate * workHours
  let savingsAmount := savingsRate * totalEarnings
  totalEarnings - savingsAmount

/-- Proves that Tom's amount available for purchases is $181.35 -/
theorem tom_purchases_amount :
  let hourlyRate : ℚ := 13/2  -- $6.50
  let workHours : ℕ := 31
  let savingsRate : ℚ := 1/10  -- 10%
  amountAvailableForPurchases hourlyRate workHours savingsRate = 36270/200  -- $181.35
  := by sorry

end NUMINAMATH_CALUDE_tom_purchases_amount_l1721_172123


namespace NUMINAMATH_CALUDE_skew_perpendicular_plane_skew_parallel_perpendicular_l1721_172147

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (skew : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- Theorem 1
theorem skew_perpendicular_plane 
  (a b : Line) (α : Plane) 
  (h1 : skew a b) 
  (h2 : perpendicular a α) : 
  ¬ perpendicular b α := by sorry

-- Theorem 2
theorem skew_parallel_perpendicular 
  (a b l : Line) (α : Plane) 
  (h1 : skew a b) 
  (h2 : parallel a α) 
  (h3 : parallel b α) 
  (h4 : perpendicular l α) : 
  perpendicularLines l a ∧ perpendicularLines l b := by sorry

end NUMINAMATH_CALUDE_skew_perpendicular_plane_skew_parallel_perpendicular_l1721_172147


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l1721_172178

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 15

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 2

/-- The total weight Terrell lifts with the original weights -/
def total_original_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of times Terrell needs to lift the new weights to match the original total weight -/
def new_lifts : ℚ := total_original_weight / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 37.5 := by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l1721_172178


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l1721_172103

theorem probability_at_least_one_correct (n : ℕ) (k : ℕ) :
  n > 0 → k > 0 →
  let p := 1 - (1 - 1 / n) ^ k
  p = 11529 / 15625 ↔ n = 5 ∧ k = 6 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l1721_172103


namespace NUMINAMATH_CALUDE_farmer_problem_solution_l1721_172166

/-- A farmer sells ducks and chickens and buys a wheelbarrow -/
def FarmerProblem (duck_price chicken_price : ℕ) (duck_sold chicken_sold : ℕ) (wheelbarrow_profit : ℕ) :=
  let total_earnings := duck_price * duck_sold + chicken_price * chicken_sold
  let wheelbarrow_cost := wheelbarrow_profit / 2
  (wheelbarrow_cost : ℚ) / total_earnings = 1 / 2

theorem farmer_problem_solution :
  FarmerProblem 10 8 2 5 60 := by sorry

end NUMINAMATH_CALUDE_farmer_problem_solution_l1721_172166


namespace NUMINAMATH_CALUDE_range_of_a_l1721_172156

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 - a ≥ 0) ∨ 
  (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) →
  ¬((∀ x ∈ Set.Icc 1 12, x^2 - a ≥ 0) ∧ 
    (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0)) →
  (-1 ≤ a ∧ a ≤ 1) ∨ a > 3 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l1721_172156


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1721_172192

/-- Given an arithmetic sequence {aₙ} where the sum of the first n terms
    is Sₙ = 3n² + 2n, prove that the general term aₙ = 6n - 1 for all
    positive integers n. -/
theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h_sum : ∀ n : ℕ, S n = 3 * n^2 + 2 * n)  -- Given condition
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence property
  : ∀ n : ℕ, n > 0 → a n = 6 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1721_172192


namespace NUMINAMATH_CALUDE_line_equation_equiv_l1721_172168

/-- The line equation in vector form -/
def line_equation (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - 8) = 0

/-- The line equation in slope-intercept form -/
def slope_intercept_form (x y : ℝ) : Prop :=
  y = (3/4) * x + (13/2)

/-- Theorem stating the equivalence of the two forms -/
theorem line_equation_equiv :
  ∀ x y : ℝ, line_equation x y ↔ slope_intercept_form x y :=
sorry

end NUMINAMATH_CALUDE_line_equation_equiv_l1721_172168


namespace NUMINAMATH_CALUDE_total_lemons_picked_l1721_172196

theorem total_lemons_picked (sally_lemons mary_lemons : ℕ) 
  (h1 : sally_lemons = 7)
  (h2 : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 := by
sorry

end NUMINAMATH_CALUDE_total_lemons_picked_l1721_172196


namespace NUMINAMATH_CALUDE_grid_value_theorem_l1721_172100

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → ℤ

/-- Checks if a sequence of 5 integers forms an arithmetic progression -/
def isArithmeticSequence (seq : Fin 5 → ℤ) : Prop :=
  ∀ i j k : Fin 5, i < j → j < k → seq j - seq i = seq k - seq j

/-- Checks if all rows and columns of a grid form arithmetic sequences -/
def isValidGrid (g : Grid) : Prop :=
  (∀ row : Fin 5, isArithmeticSequence (λ col => g row col)) ∧
  (∀ col : Fin 5, isArithmeticSequence (λ row => g row col))

theorem grid_value_theorem (g : Grid) :
  isValidGrid g →
  g 1 1 = 74 →
  g 2 4 = 186 →
  g 3 2 = 103 →
  g 4 0 = 0 →
  g 0 3 = 142 := by
  sorry

#check grid_value_theorem

end NUMINAMATH_CALUDE_grid_value_theorem_l1721_172100


namespace NUMINAMATH_CALUDE_weekend_sleep_calculation_l1721_172119

/-- Calculates the number of hours slept during weekends per day, given the total weekly sleep and weekday sleep hours. -/
def weekend_sleep_hours (total_weekly_sleep : ℕ) (weekday_sleep : ℕ) : ℚ :=
  ((total_weekly_sleep - (weekday_sleep * 5)) : ℚ) / 2

/-- Theorem stating that given 51 hours of total weekly sleep and 7 hours of sleep each weekday, 
    the number of hours slept each day during weekends is 8. -/
theorem weekend_sleep_calculation :
  weekend_sleep_hours 51 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_weekend_sleep_calculation_l1721_172119


namespace NUMINAMATH_CALUDE_factor_polynomial_l1721_172189

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = -27 * x^5 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1721_172189


namespace NUMINAMATH_CALUDE_point_on_line_l1721_172186

/-- Given three points in the plane, this function checks if they are collinear -/
def are_collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that the point (14,7) lies on the line passing through (2,1) and (10,5) -/
theorem point_on_line : are_collinear 2 1 10 5 14 7 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1721_172186


namespace NUMINAMATH_CALUDE_gaskets_sold_l1721_172120

/-- Calculates the total cost of gasket packages --/
def totalCost (packages : ℕ) : ℚ :=
  if packages ≤ 10 then
    25 * packages
  else
    250 + 20 * (packages - 10)

/-- Proves that 65 packages of gaskets were sold given the conditions --/
theorem gaskets_sold : ∃ (packages : ℕ), packages > 10 ∧ totalCost packages = 1340 := by
  sorry

#eval totalCost 65

end NUMINAMATH_CALUDE_gaskets_sold_l1721_172120


namespace NUMINAMATH_CALUDE_fir_trees_not_adjacent_probability_l1721_172198

def num_pine : ℕ := 5
def num_cedar : ℕ := 6
def num_fir : ℕ := 7
def total_trees : ℕ := num_pine + num_cedar + num_fir

def valid_arrangements : ℕ := Nat.choose (num_pine + num_cedar + 1) num_fir
def total_arrangements : ℕ := Nat.choose total_trees num_fir

theorem fir_trees_not_adjacent_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 40 := by sorry

end NUMINAMATH_CALUDE_fir_trees_not_adjacent_probability_l1721_172198


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l1721_172130

theorem quadratic_form_k_value (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l1721_172130


namespace NUMINAMATH_CALUDE_second_machine_copies_per_minute_l1721_172199

/-- 
Given two copy machines working at constant rates, where the first machine makes 35 copies per minute,
and together they make 3300 copies in 30 minutes, prove that the second machine makes 75 copies per minute.
-/
theorem second_machine_copies_per_minute 
  (rate1 : ℕ) 
  (rate2 : ℕ) 
  (total_time : ℕ) 
  (total_copies : ℕ) 
  (h1 : rate1 = 35)
  (h2 : total_time = 30)
  (h3 : total_copies = 3300)
  (h4 : rate1 * total_time + rate2 * total_time = total_copies) : 
  rate2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_second_machine_copies_per_minute_l1721_172199


namespace NUMINAMATH_CALUDE_prime_between_squares_l1721_172170

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ x : ℕ, p = x^2 + 5 ∧ p + 9 = (x + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_prime_between_squares_l1721_172170


namespace NUMINAMATH_CALUDE_count_divisible_numbers_main_result_l1721_172116

theorem count_divisible_numbers (n : ℕ) (m : ℕ) : 
  (Finset.filter (fun k => (k^2 - 1) % m = 0) (Finset.range (n + 1))).card = 4 * (n / m) :=
by
  sorry

theorem main_result : 
  (Finset.filter (fun k => (k^2 - 1) % 485 = 0) (Finset.range 485001)).card = 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_main_result_l1721_172116


namespace NUMINAMATH_CALUDE_conditional_probability_b_given_a_and_c_l1721_172133

-- Define the sample space and probability measure
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)

-- Define events as measurable sets
variable (a b c : Set Ω)

-- Define probabilities
variable (pa pb pc pab pac pbc pabc : ℝ)

-- State the theorem
theorem conditional_probability_b_given_a_and_c
  (h_pa : P a = pa)
  (h_pb : P b = pb)
  (h_pc : P c = pc)
  (h_pab : P (a ∩ b) = pab)
  (h_pac : P (a ∩ c) = pac)
  (h_pbc : P (b ∩ c) = pbc)
  (h_pabc : P (a ∩ b ∩ c) = pabc)
  (h_pa_val : pa = 5/23)
  (h_pb_val : pb = 7/23)
  (h_pc_val : pc = 1/23)
  (h_pab_val : pab = 2/23)
  (h_pac_val : pac = 1/23)
  (h_pbc_val : pbc = 1/23)
  (h_pabc_val : pabc = 1/23)
  : P (b ∩ (a ∩ c)) / P (a ∩ c) = 1 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_b_given_a_and_c_l1721_172133


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1721_172153

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}

theorem complement_of_A_wrt_U : 
  {x ∈ U | x ∉ A} = {4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1721_172153


namespace NUMINAMATH_CALUDE_boys_average_score_l1721_172150

theorem boys_average_score (num_boys num_girls : ℕ) (girls_avg class_avg : ℝ) :
  num_boys = 12 →
  num_girls = 4 →
  girls_avg = 92 →
  class_avg = 86 →
  (num_boys * (class_avg * (num_boys + num_girls) - num_girls * girls_avg)) / (num_boys * (num_boys + num_girls)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_boys_average_score_l1721_172150


namespace NUMINAMATH_CALUDE_solution_in_quadrant_IV_l1721_172152

/-- Given a system of equations x + 2y = 4 and kx - y = 1, where k is a constant,
    the solution (x, y) is in Quadrant IV if and only if -1/2 < k < 2 -/
theorem solution_in_quadrant_IV (k : ℝ) : 
  (∃ x y : ℝ, x + 2*y = 4 ∧ k*x - y = 1 ∧ x > 0 ∧ y < 0) ↔ -1/2 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_IV_l1721_172152


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1721_172138

/-- 
Given a parabola y = ax^2 + 6 that is tangent to the line y = 2x - 3,
prove that the value of the constant a is 1/9.
-/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 6 = 2 * x - 3 ∧ 
   ∀ y : ℝ, y ≠ x → a * y^2 + 6 ≠ 2 * y - 3) →
  a = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1721_172138


namespace NUMINAMATH_CALUDE_improved_milk_production_l1721_172117

/-- Given initial milk production parameters and an efficiency increase,
    calculate the new milk production for a different number of cows and days. -/
theorem improved_milk_production
  (a b c d e : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
  (h_initial : b / (a * c) = initial_rate)
  (h_efficiency_increase : new_rate = initial_rate * 1.2) :
  new_rate * d * e = (1.2 * b * d * e) / (a * c) :=
sorry

end NUMINAMATH_CALUDE_improved_milk_production_l1721_172117
