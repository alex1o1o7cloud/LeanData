import Mathlib

namespace NUMINAMATH_CALUDE_plane_seats_count_l485_48513

theorem plane_seats_count : 
  ∀ (total_seats : ℕ),
  (30 : ℕ) + (total_seats / 5 : ℕ) + (total_seats - (30 + total_seats / 5) : ℕ) = total_seats →
  total_seats = 50 := by
sorry

end NUMINAMATH_CALUDE_plane_seats_count_l485_48513


namespace NUMINAMATH_CALUDE_micahs_strawberries_l485_48539

def strawberries_for_mom (picked : ℕ) (eaten : ℕ) : ℕ :=
  picked - eaten

theorem micahs_strawberries :
  strawberries_for_mom (2 * 12) 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_micahs_strawberries_l485_48539


namespace NUMINAMATH_CALUDE_max_distance_between_paths_l485_48535

theorem max_distance_between_paths : 
  ∃ (C : ℝ), C = 3 * Real.sqrt 3 ∧ 
  ∀ (t : ℝ), 
    Real.sqrt ((t - (t - 5))^2 + (Real.sin t - Real.cos (t - 5))^2) ≤ C :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_paths_l485_48535


namespace NUMINAMATH_CALUDE_positive_expression_l485_48551

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c) + a * (b^2 + c^2 - b*c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l485_48551


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l485_48563

theorem reciprocal_sum_fractions : 
  (1 / (1/3 + 1/4 - 1/12) : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l485_48563


namespace NUMINAMATH_CALUDE_fraction_value_l485_48523

/-- Given a, b, c, d are real numbers satisfying certain relationships,
    prove that (a * c) / (b * d) = 15 -/
theorem fraction_value (a b c d : ℝ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * c) 
    (h3 : c = 5 * d) 
    (h4 : b ≠ 0) 
    (h5 : d ≠ 0) : 
  (a * c) / (b * d) = 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l485_48523


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l485_48567

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Main theorem: If f is increasing and f(2m) > f(-m+9), then m > 3 -/
theorem increasing_function_inequality (f : ℝ → ℝ) (m : ℝ) 
    (h_incr : IsIncreasing f) (h_ineq : f (2 * m) > f (-m + 9)) : 
    m > 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l485_48567


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l485_48504

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2) = π * (r₁^2 - r₂^2) := by sorry

/-- The area of the ring between two concentric circles with radii 15 and 9 -/
theorem area_of_specific_ring : 
  (π * 15^2 - π * 9^2) = 144 * π := by sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l485_48504


namespace NUMINAMATH_CALUDE_unused_signs_l485_48524

theorem unused_signs (total_signs : Nat) (used_signs : Nat) (additional_codes : Nat) : 
  total_signs = 424 →
  used_signs = 422 →
  additional_codes = 1688 →
  total_signs ^ 2 - used_signs ^ 2 = additional_codes →
  total_signs - used_signs = 2 :=
by sorry

end NUMINAMATH_CALUDE_unused_signs_l485_48524


namespace NUMINAMATH_CALUDE_product_sum_equation_l485_48581

theorem product_sum_equation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 + 1) = 107 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equation_l485_48581


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l485_48573

-- Define the quadratic function
def f (a c : ℝ) (x : ℝ) := a * x^2 + 2 * x + c

-- Define the solution set
def solution_set (a c : ℝ) := {x : ℝ | x < -1 ∨ x > 2}

-- State the theorem
theorem quadratic_inequality_properties
  (a c : ℝ)
  (h : ∀ x, f a c x < 0 ↔ x ∈ solution_set a c) :
  (a + c = 2) ∧
  (c^(1/a) = 1/2) ∧
  (∃! y, y ∈ {x : ℝ | x^2 - 2*a*x + c = 0}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l485_48573


namespace NUMINAMATH_CALUDE_impossible_arrangement_l485_48586

theorem impossible_arrangement : ¬ ∃ (A B C : ℕ), 
  (A + B = 45) ∧ 
  (3 * A + B = 6 * C) ∧ 
  (A ≥ 0) ∧ (B ≥ 0) ∧ (C > 0) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l485_48586


namespace NUMINAMATH_CALUDE_root_product_theorem_l485_48564

theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 2 = 0) →
  (y₂^5 - y₂^3 + 2 = 0) →
  (y₃^5 - y₃^3 + 2 = 0) →
  (y₄^5 - y₄^3 + 2 = 0) →
  (y₅^5 - y₅^3 + 2 = 0) →
  (y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = 104 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l485_48564


namespace NUMINAMATH_CALUDE_ellipse_tangent_to_circle_l485_48553

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

def ellipse (p : ℝ × ℝ) : Prop :=
  p.1^2 / 4 + p.2^2 / 2 = 1

def on_line_y_eq_neg_2 (p : ℝ × ℝ) : Prop :=
  p.2 = -2

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def tangent_to_circle (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ l ∧ p.1^2 + p.2^2 = 2 ∧
    ∀ q, q ∈ l → q.1^2 + q.2^2 ≥ 2

theorem ellipse_tangent_to_circle :
  ∀ E F : ℝ × ℝ,
    ellipse E →
    on_line_y_eq_neg_2 F →
    perpendicular E F →
    tangent_to_circle {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • E + t • F} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_to_circle_l485_48553


namespace NUMINAMATH_CALUDE_quadratic_vertex_l485_48527

/-- The quadratic function f(x) = 2(x-1)^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 + 5

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (1, 5)

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-1)^2 + 5 is (1, 5) -/
theorem quadratic_vertex : 
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l485_48527


namespace NUMINAMATH_CALUDE_butter_cost_l485_48514

theorem butter_cost (initial_amount spent_on_bread spent_on_juice remaining_amount : ℝ)
  (h1 : initial_amount = 15)
  (h2 : remaining_amount = 6)
  (h3 : spent_on_bread = 2)
  (h4 : spent_on_juice = 2 * spent_on_bread)
  : initial_amount - remaining_amount - spent_on_bread - spent_on_juice = 3 := by
  sorry

end NUMINAMATH_CALUDE_butter_cost_l485_48514


namespace NUMINAMATH_CALUDE_sophies_purchase_amount_l485_48531

/-- Calculates the total amount Sophie spends on her purchase --/
def sophies_purchase (cupcake_price : ℚ) (doughnut_price : ℚ) (pie_price : ℚ)
  (cookie_price : ℚ) (chocolate_price : ℚ) (soda_price : ℚ) (gum_price : ℚ)
  (chips_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let subtotal := 5 * cupcake_price + 6 * doughnut_price + 4 * pie_price +
    15 * cookie_price + 8 * chocolate_price + 12 * soda_price +
    3 * gum_price + 10 * chips_price
  let discounted_total := subtotal * (1 - discount_rate)
  let tax_amount := discounted_total * tax_rate
  discounted_total + tax_amount

/-- Theorem stating that Sophie's total purchase amount is $69.45 --/
theorem sophies_purchase_amount :
  sophies_purchase 2 1 2 (6/10) (3/2) (6/5) (4/5) (11/10) (1/10) (6/100) = (6945/100) := by
  sorry

end NUMINAMATH_CALUDE_sophies_purchase_amount_l485_48531


namespace NUMINAMATH_CALUDE_range_of_a_l485_48521

/-- The range of a given the conditions in the problem -/
theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 4*a*x + 3*a^2 < 0 → |x - 3| > 1) ∧ 
  (∃ x, |x - 3| > 1 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) ∧ 
  (a > 0) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry


end NUMINAMATH_CALUDE_range_of_a_l485_48521


namespace NUMINAMATH_CALUDE_g_properties_l485_48590

noncomputable def g (x : ℝ) : ℝ := (4 * Real.sin x ^ 4 + 7 * Real.cos x ^ 2) / (4 * Real.cos x ^ 4 + Real.sin x ^ 2)

theorem g_properties :
  (∀ k : ℤ, g (Real.pi / 3 + k * Real.pi) = 4 ∧ g (-Real.pi / 3 + k * Real.pi) = 4 ∧ g (Real.pi / 2 + k * Real.pi) = 4) ∧
  (∀ x : ℝ, g x ≥ 7 / 4) ∧
  (∀ x : ℝ, g x ≤ 63 / 15) ∧
  (∃ x : ℝ, g x = 7 / 4) ∧
  (∃ x : ℝ, g x = 63 / 15) := by
  sorry

end NUMINAMATH_CALUDE_g_properties_l485_48590


namespace NUMINAMATH_CALUDE_baga_answer_variability_l485_48574

/-- Represents a BAGA problem -/
structure BAGAProblem where
  conditions : Set String
  approach : String

/-- Represents the answer to a BAGA problem -/
structure BAGAAnswer where
  value : String

/-- Function that solves a BAGA problem -/
noncomputable def solveBagaProblem (problem : BAGAProblem) : BAGAAnswer :=
  sorry

/-- Theorem stating that small variations in BAGA problems can lead to different answers -/
theorem baga_answer_variability 
  (p1 p2 : BAGAProblem) 
  (h_small_diff : p1.conditions ≠ p2.conditions ∨ p1.approach ≠ p2.approach) : 
  ∃ (a1 a2 : BAGAAnswer), solveBagaProblem p1 = a1 ∧ solveBagaProblem p2 = a2 ∧ a1 ≠ a2 :=
sorry

end NUMINAMATH_CALUDE_baga_answer_variability_l485_48574


namespace NUMINAMATH_CALUDE_white_tulips_multiple_of_seven_l485_48516

/-- The number of red tulips -/
def red_tulips : ℕ := 91

/-- The number of identical bouquets that can be made -/
def num_bouquets : ℕ := 7

/-- The number of white tulips -/
def white_tulips : ℕ := sorry

/-- Proposition stating that the number of white tulips is a multiple of 7 -/
theorem white_tulips_multiple_of_seven :
  ∃ k : ℕ, white_tulips = 7 * k ∧ red_tulips % num_bouquets = 0 :=
sorry

end NUMINAMATH_CALUDE_white_tulips_multiple_of_seven_l485_48516


namespace NUMINAMATH_CALUDE_value_of_P_closed_under_multiplication_l485_48543

/-- The polynomial P(x, y) = 2x^2 - 6xy + 5y^2 -/
def P (x y : ℤ) : ℤ := 2*x^2 - 6*x*y + 5*y^2

/-- A number is a value of P if it can be expressed as P(b, c) for some integers b and c -/
def is_value_of_P (a : ℤ) : Prop := ∃ b c : ℤ, P b c = a

/-- If r and s are values of P, then rs is also a value of P -/
theorem value_of_P_closed_under_multiplication (r s : ℤ) 
  (hr : is_value_of_P r) (hs : is_value_of_P s) : 
  is_value_of_P (r * s) := by
  sorry

end NUMINAMATH_CALUDE_value_of_P_closed_under_multiplication_l485_48543


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l485_48528

theorem arithmetic_sequence_sum : 
  ∀ (a₁ aₙ d n : ℕ) (S : ℕ),
    a₁ = 1 →                   -- First term is 1
    aₙ = 25 →                  -- Last term is 25
    d = 2 →                    -- Common difference is 2
    aₙ = a₁ + (n - 1) * d →    -- Formula for the nth term of an arithmetic sequence
    S = n * (a₁ + aₙ) / 2 →    -- Formula for the sum of an arithmetic sequence
    S = 169 :=                 -- The sum is 169
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l485_48528


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l485_48589

def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : inversely_proportional x₁ y₁)
  (h2 : inversely_proportional x₂ y₂)
  (h3 : x₁ = 5)
  (h4 : y₁ = 15)
  (h5 : y₂ = 30) :
  x₂ = 5/2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l485_48589


namespace NUMINAMATH_CALUDE_linear_inequality_condition_l485_48503

theorem linear_inequality_condition (m : ℝ) : 
  (|m - 3| = 1 ∧ m - 4 ≠ 0) ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_linear_inequality_condition_l485_48503


namespace NUMINAMATH_CALUDE_sum_of_z_values_l485_48537

-- Define the function f
def f (x : ℝ) : ℝ := (4*x)^2 - 3*(4*x) + 2

-- State the theorem
theorem sum_of_z_values (f : ℝ → ℝ) : 
  (f = λ x => (4*x)^2 - 3*(4*x) + 2) → 
  (∃ z₁ z₂ : ℝ, f z₁ = 9 ∧ f z₂ = 9 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 3/16) := by
  sorry


end NUMINAMATH_CALUDE_sum_of_z_values_l485_48537


namespace NUMINAMATH_CALUDE_compute_expression_l485_48570

theorem compute_expression : (75 * 2424 + 25 * 2424) / 2 = 121200 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l485_48570


namespace NUMINAMATH_CALUDE_mean_score_is_94_5_l485_48571

/-- Represents the score distribution of students -/
structure ScoreDistribution where
  score : ℕ
  num_students : ℕ

/-- Calculates the mean score given a list of score distributions -/
def mean_score (distributions : List ScoreDistribution) (total_students : ℕ) : ℚ :=
  let total_score := (distributions.map (λ d => d.score * d.num_students)).sum
  total_score / total_students

/-- The given score distribution from the problem -/
def exam_distribution : List ScoreDistribution := [
  ⟨120, 12⟩,
  ⟨110, 19⟩,
  ⟨100, 33⟩,
  ⟨90, 30⟩,
  ⟨75, 15⟩,
  ⟨65, 9⟩,
  ⟨50, 2⟩
]

/-- The total number of students -/
def total_students : ℕ := 120

/-- Theorem stating that the mean score is 94.5 -/
theorem mean_score_is_94_5 :
  mean_score exam_distribution total_students = 945 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_score_is_94_5_l485_48571


namespace NUMINAMATH_CALUDE_equation_solutions_l485_48587

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 4 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 3 * x^2 - 1 = 2 * x ↔ x = 1 ∨ x = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l485_48587


namespace NUMINAMATH_CALUDE_george_score_l485_48501

theorem george_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) : 
  n = 20 → 
  avg_without = 75 → 
  avg_with = 76 → 
  (n - 1) * avg_without + 95 = n * avg_with :=
by
  sorry

end NUMINAMATH_CALUDE_george_score_l485_48501


namespace NUMINAMATH_CALUDE_function_property_implies_k_equals_8_l485_48556

/-- Given a function f: ℝ → ℝ satisfying certain properties, prove that k = 8 -/
theorem function_property_implies_k_equals_8 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y, f (x + y) = f x + f y + k * x * y - 2)
  (h3 : f 7 = 163) :
  k = 8 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_k_equals_8_l485_48556


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l485_48510

theorem opposite_of_negative_three :
  -(- 3) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l485_48510


namespace NUMINAMATH_CALUDE_inequality_proof_l485_48526

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x / y + y / z + z / x) / 3 ≥ 1) ∧
  (x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3) ∧
  (x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l485_48526


namespace NUMINAMATH_CALUDE_sum_odd_when_sum_of_squares_odd_l485_48505

theorem sum_odd_when_sum_of_squares_odd (n m : ℤ) (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_when_sum_of_squares_odd_l485_48505


namespace NUMINAMATH_CALUDE_bird_multiple_l485_48534

theorem bird_multiple : ∃ x : ℝ, x * 20 + 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bird_multiple_l485_48534


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l485_48545

theorem choose_four_from_seven : 
  Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l485_48545


namespace NUMINAMATH_CALUDE_cat_adoption_rate_is_25_percent_l485_48559

def initial_dogs : ℕ := 30
def initial_cats : ℕ := 28
def initial_lizards : ℕ := 20
def dog_adoption_rate : ℚ := 1/2
def lizard_adoption_rate : ℚ := 1/5
def new_pets : ℕ := 13
def total_pets_after_month : ℕ := 65

theorem cat_adoption_rate_is_25_percent :
  let dogs_adopted := (initial_dogs : ℚ) * dog_adoption_rate
  let lizards_adopted := (initial_lizards : ℚ) * lizard_adoption_rate
  let remaining_dogs := initial_dogs - dogs_adopted.floor
  let remaining_lizards := initial_lizards - lizards_adopted.floor
  let remaining_pets := remaining_dogs + remaining_lizards + new_pets
  let remaining_cats := total_pets_after_month - remaining_pets
  let cats_adopted := initial_cats - remaining_cats
  (cats_adopted : ℚ) / initial_cats = 1/4 := by
    sorry

end NUMINAMATH_CALUDE_cat_adoption_rate_is_25_percent_l485_48559


namespace NUMINAMATH_CALUDE_will_had_28_bottles_l485_48546

/-- The number of bottles Will had -/
def bottles : ℕ := sorry

/-- The number of days the bottles would last -/
def days : ℕ := 4

/-- The number of bottles Will would drink per day -/
def bottles_per_day : ℕ := 7

/-- Theorem stating that Will had 28 bottles -/
theorem will_had_28_bottles : bottles = 28 := by
  sorry

end NUMINAMATH_CALUDE_will_had_28_bottles_l485_48546


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l485_48548

theorem square_sum_equals_one (a b : ℝ) (h : a + b = -1) : a^2 + b^2 + 2*a*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l485_48548


namespace NUMINAMATH_CALUDE_polygon_with_five_triangles_is_heptagon_l485_48569

/-- The number of triangles formed by diagonals from one vertex in an n-sided polygon -/
def triangles_from_diagonals (n : ℕ) : ℕ := n - 2

theorem polygon_with_five_triangles_is_heptagon (n : ℕ) :
  (n ≥ 3) → (triangles_from_diagonals n = 5) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_five_triangles_is_heptagon_l485_48569


namespace NUMINAMATH_CALUDE_train_length_l485_48533

/-- The length of a train given its speed and time to pass a point -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 216) (h2 : time = 6) :
  speed * (5 / 18) * time = 360 :=
sorry

end NUMINAMATH_CALUDE_train_length_l485_48533


namespace NUMINAMATH_CALUDE_equal_numbers_l485_48507

theorem equal_numbers (x : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, (x i)^3 + x (i + 1) = (x (i + 1))^3 + x (i + 2)) : 
  ∀ i j : Fin 100, x i = x j := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_l485_48507


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l485_48566

/-- The sum of digits of a number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatDigit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : 
  let a := repeatDigit 8 2000
  let b := repeatDigit 5 2000
  sumOfDigits (9 * a * b) = 18005 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l485_48566


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l485_48554

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem states that for a geometric sequence satisfying certain conditions, 
    the sum of its 2nd and 8th terms equals 9. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geo : IsGeometricSequence a) 
  (h_prod : a 3 * a 7 = 8)
  (h_sum : a 4 + a 6 = 6) : 
  a 2 + a 8 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l485_48554


namespace NUMINAMATH_CALUDE_oscar_fish_count_l485_48565

/-- Represents the initial number of Oscar fish in Danny's fish tank. -/
def initial_oscar_fish : ℕ := 58

/-- Theorem stating that the initial number of Oscar fish was 58. -/
theorem oscar_fish_count :
  let initial_guppies : ℕ := 94
  let initial_angelfish : ℕ := 76
  let initial_tiger_sharks : ℕ := 89
  let sold_guppies : ℕ := 30
  let sold_angelfish : ℕ := 48
  let sold_tiger_sharks : ℕ := 17
  let sold_oscar_fish : ℕ := 24
  let remaining_fish : ℕ := 198
  initial_oscar_fish = 
    remaining_fish - 
    ((initial_guppies - sold_guppies) + 
     (initial_angelfish - sold_angelfish) + 
     (initial_tiger_sharks - sold_tiger_sharks)) + 
    sold_oscar_fish :=
by sorry

end NUMINAMATH_CALUDE_oscar_fish_count_l485_48565


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l485_48541

theorem geometric_arithmetic_sequence_sum (x y : ℝ) :
  0 < x ∧ 0 < y ∧
  (1 : ℝ) * x = x * y ∧  -- Geometric sequence condition
  y - x = 3 - y →        -- Arithmetic sequence condition
  x + y = 15/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l485_48541


namespace NUMINAMATH_CALUDE_investment_profit_distribution_l485_48599

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  total_contribution : ℝ
  a_duration : ℝ
  b_duration : ℝ
  a_final : ℝ
  b_final : ℝ
  a_contribution : ℝ
  b_contribution : ℝ

/-- Theorem stating that the given contributions satisfy the profit distribution -/
theorem investment_profit_distribution (investment : BusinessInvestment)
  (h1 : investment.total_contribution = 1500)
  (h2 : investment.a_duration = 3)
  (h3 : investment.b_duration = 4)
  (h4 : investment.a_final = 690)
  (h5 : investment.b_final = 1080)
  (h6 : investment.a_contribution = 600)
  (h7 : investment.b_contribution = 900)
  (h8 : investment.a_contribution + investment.b_contribution = investment.total_contribution) :
  (investment.a_final - investment.a_contribution) / (investment.b_final - investment.b_contribution) =
  (investment.a_duration * investment.a_contribution) / (investment.b_duration * investment.b_contribution) :=
by sorry

end NUMINAMATH_CALUDE_investment_profit_distribution_l485_48599


namespace NUMINAMATH_CALUDE_functional_equation_1_bijective_functional_equation_2_neither_functional_equation_3_neither_functional_equation_4_neither_l485_48562

-- 1. f(x+f(y))=2f(x)+y is bijective
theorem functional_equation_1_bijective (f : ℝ → ℝ) :
  (∀ x y, f (x + f y) = 2 * f x + y) → Function.Bijective f :=
sorry

-- 2. f(f(x))=0 is neither injective nor surjective
theorem functional_equation_2_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = 0) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 3. f(f(x))=sin(x) is neither injective nor surjective
theorem functional_equation_3_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = Real.sin x) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 4. f(x+y)=f(x)f(y) is neither injective nor surjective
theorem functional_equation_4_neither (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x * f y) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_1_bijective_functional_equation_2_neither_functional_equation_3_neither_functional_equation_4_neither_l485_48562


namespace NUMINAMATH_CALUDE_billboard_area_l485_48547

/-- The area of a rectangular billboard with perimeter 46 feet and width 8 feet is 120 square feet. -/
theorem billboard_area (perimeter width : ℝ) (h1 : perimeter = 46) (h2 : width = 8) :
  let length := (perimeter - 2 * width) / 2
  width * length = 120 :=
by sorry

end NUMINAMATH_CALUDE_billboard_area_l485_48547


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l485_48575

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ ∀ x, f x ≠ 0 := by sorry

theorem cubic_equation_negation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  apply negation_of_existence

end NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l485_48575


namespace NUMINAMATH_CALUDE_smallest_result_l485_48555

def number_set : Finset ℕ := {3, 4, 7, 11, 13, 14}

def is_prime_greater_than_10 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 10

def valid_triple (a b c : ℕ) : Prop :=
  a ∈ number_set ∧ b ∈ number_set ∧ c ∈ number_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (is_prime_greater_than_10 a ∨ is_prime_greater_than_10 b ∨ is_prime_greater_than_10 c)

def process_result (a b c : ℕ) : ℕ :=
  (a + b) * c

theorem smallest_result :
  ∀ a b c : ℕ, valid_triple a b c →
    77 ≤ min (process_result a b c) (min (process_result a c b) (process_result b c a)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l485_48555


namespace NUMINAMATH_CALUDE_cube_of_product_l485_48525

theorem cube_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l485_48525


namespace NUMINAMATH_CALUDE_inequality_implication_l485_48584

theorem inequality_implication (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l485_48584


namespace NUMINAMATH_CALUDE_green_flash_count_l485_48506

def total_time : ℕ := 671
def green_interval : ℕ := 3
def red_interval : ℕ := 5
def blue_interval : ℕ := 7

def green_flashes : ℕ := total_time / green_interval
def green_red_flashes : ℕ := total_time / (Nat.lcm green_interval red_interval)
def green_blue_flashes : ℕ := total_time / (Nat.lcm green_interval blue_interval)
def all_color_flashes : ℕ := total_time / (Nat.lcm green_interval (Nat.lcm red_interval blue_interval))

theorem green_flash_count : 
  green_flashes - green_red_flashes - green_blue_flashes + all_color_flashes = 154 := by
  sorry

end NUMINAMATH_CALUDE_green_flash_count_l485_48506


namespace NUMINAMATH_CALUDE_center_sum_is_seven_l485_48595

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 8*y - 15

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - (6*h + 8*k - 15))

theorem center_sum_is_seven :
  ∃ h k, CircleCenter h k CircleEquation ∧ h + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_center_sum_is_seven_l485_48595


namespace NUMINAMATH_CALUDE_cube_coloring_theorem_l485_48597

/-- Represents the symmetry group of a cube -/
def CubeSymmetryGroup : Type := Unit

/-- The order of the cube symmetry group -/
def symmetryGroupOrder : ℕ := 24

/-- The total number of ways to color a cube with 6 colors without considering rotations -/
def totalColorings : ℕ := 720

/-- The number of distinct colorings of a cube with 6 colors, considering rotational symmetries -/
def distinctColorings : ℕ := totalColorings / symmetryGroupOrder

theorem cube_coloring_theorem :
  distinctColorings = 30 :=
sorry

end NUMINAMATH_CALUDE_cube_coloring_theorem_l485_48597


namespace NUMINAMATH_CALUDE_tip_difference_proof_l485_48518

/-- The difference between a good tip (20%) and a bad tip (5%) on a $26 bill is 390 cents. -/
theorem tip_difference_proof : 
  let bill : ℚ := 26
  let bad_tip_rate : ℚ := 5 / 100
  let good_tip_rate : ℚ := 20 / 100
  (good_tip_rate * bill - bad_tip_rate * bill) * 100 = 390 := by
sorry

end NUMINAMATH_CALUDE_tip_difference_proof_l485_48518


namespace NUMINAMATH_CALUDE_sin_negative_ninety_degrees_l485_48511

theorem sin_negative_ninety_degrees :
  Real.sin (- π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_ninety_degrees_l485_48511


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l485_48578

theorem largest_prime_factors_difference (n : Nat) (h : n = 171689) : 
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧ 
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
    p ∣ n ∧ 
    q ∣ n ∧ 
    p - q = 282 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l485_48578


namespace NUMINAMATH_CALUDE_segment_intersection_theorem_l485_48542

/-- Represents a line in the real plane -/
structure Line where
  -- Add necessary fields

/-- Represents a line segment in the real plane -/
structure Segment where
  -- Add necessary fields

/-- Predicate to check if a line intersects a segment -/
def intersects (l : Line) (s : Segment) : Prop :=
  sorry

/-- Predicate to check if segments are concurrent -/
def concurrent (segments : List Segment) : Prop :=
  sorry

theorem segment_intersection_theorem
  (n : ℕ)
  (segments : List Segment)
  (h_concurrent : concurrent segments)
  (h_count : segments.length = n)
  (h_triple_intersection : ∀ (s1 s2 s3 : Segment),
    s1 ∈ segments → s2 ∈ segments → s3 ∈ segments →
    ∃ (l : Line), intersects l s1 ∧ intersects l s2 ∧ intersects l s3) :
  ∃ (l : Line), ∀ (s : Segment), s ∈ segments → intersects l s :=
sorry

end NUMINAMATH_CALUDE_segment_intersection_theorem_l485_48542


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l485_48576

theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 113 →
  jake_weight - 33 = 2 * sister_weight →
  jake_weight + sister_weight = 153 := by
sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l485_48576


namespace NUMINAMATH_CALUDE_mk97_check_one_l485_48580

theorem mk97_check_one (a : ℝ) : 
  (a = 1) ↔ (a ≠ 2 * a ∧ 
             ∃ x : ℝ, x^2 + 2*a*x + a = 0 ∧ 
             ∀ y : ℝ, y^2 + 2*a*y + a = 0 → y = x) := by
  sorry

end NUMINAMATH_CALUDE_mk97_check_one_l485_48580


namespace NUMINAMATH_CALUDE_water_depth_in_tank_l485_48568

/-- Represents a horizontally placed cylindrical water tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the possible depths of water in a cylindrical tank -/
def water_depths (tank : CylindricalTank) (water_surface_area : ℝ) : Set ℝ :=
  sorry

/-- Theorem stating the depths of water in the given cylindrical tank -/
theorem water_depth_in_tank (tank : CylindricalTank) 
  (h1 : tank.length = 12)
  (h2 : tank.diameter = 8)
  (h3 : water_surface_area = 48) :
  water_depths tank water_surface_area = {4 - 2 * Real.sqrt 3, 4 + 2 * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_water_depth_in_tank_l485_48568


namespace NUMINAMATH_CALUDE_coin_flip_probability_l485_48560

theorem coin_flip_probability : 
  let n : ℕ := 12  -- Total number of coins
  let k : ℕ := 3   -- Maximum number of heads we're interested in
  let favorable_outcomes : ℕ := (Finset.range (k + 1)).sum (λ i => Nat.choose n i)
  let total_outcomes : ℕ := 2^n
  (favorable_outcomes : ℚ) / total_outcomes = 299 / 4096 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l485_48560


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_l485_48515

theorem geometric_sequence_roots (a b : ℝ) : 
  (∃ x₁ x₄ : ℝ, x₁ ≠ x₄ ∧ x₁^2 - 9*x₁ + 2^a = 0 ∧ x₄^2 - 9*x₄ + 2^a = 0) →
  (∃ x₂ x₃ : ℝ, x₂ ≠ x₃ ∧ x₂^2 - 6*x₂ + 2^b = 0 ∧ x₃^2 - 6*x₃ + 2^b = 0) →
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ * 2 = x₂ ∧ x₂ * 2 = x₃ ∧ x₃ * 2 = x₄ ∧
    x₁^2 - 9*x₁ + 2^a = 0 ∧ x₄^2 - 9*x₄ + 2^a = 0 ∧
    x₂^2 - 6*x₂ + 2^b = 0 ∧ x₃^2 - 6*x₃ + 2^b = 0) →
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_l485_48515


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l485_48577

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + a*x + 1

-- Define what it means for x to be a fixed point of f
def is_fixed_point (a x : ℝ) : Prop := f a x = x

-- Theorem statement
theorem no_fixed_points_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬(is_fixed_point a x)) ↔ (-1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l485_48577


namespace NUMINAMATH_CALUDE_x_twelfth_power_is_one_l485_48517

theorem x_twelfth_power_is_one (x : ℂ) (h : x + 1/x = -1) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_is_one_l485_48517


namespace NUMINAMATH_CALUDE_line_passes_through_point_l485_48552

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (t m x y : ℝ) : Prop := x = t*y + m

-- Define point P
def point_P : ℝ × ℝ := (-2, 0)

-- Define the condition that l is not vertical to x-axis
def not_vertical (t : ℝ) : Prop := t ≠ 0

-- Define the bisection condition
def bisects_angle (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  y₁ / (x₁ + 2) + y₂ / (x₂ + 2) = 0

-- Main theorem
theorem line_passes_through_point :
  ∀ (t m : ℝ) (A B : ℝ × ℝ),
  not_vertical t →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line_l t m A.1 A.2 →
  line_l t m B.1 B.2 →
  A ≠ B →
  bisects_angle A B →
  ∃ (x : ℝ), line_l t m x 0 ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l485_48552


namespace NUMINAMATH_CALUDE_parabola_through_point_l485_48596

theorem parabola_through_point (a b c : ℤ) : 
  5 = a * 2^2 + b * 2 + c → 2 * a + b - c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l485_48596


namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l485_48561

theorem unique_digit_arrangement : ∃! (a b c d e : ℕ),
  (0 < a ∧ a ≤ 9) ∧
  (0 < b ∧ b ≤ 9) ∧
  (0 < c ∧ c ≤ 9) ∧
  (0 < d ∧ d ≤ 9) ∧
  (0 < e ∧ e ≤ 9) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a + b = (c + d + e) / 7 ∧
  a + c = (b + d + e) / 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_arrangement_l485_48561


namespace NUMINAMATH_CALUDE_ratio_equality_l485_48502

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l485_48502


namespace NUMINAMATH_CALUDE_circle_trajectory_l485_48592

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 81
def F₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Define the center and radius of F₁
def F₁_center : ℝ × ℝ := (-3, 0)
def F₁_radius : ℝ := 9

-- Define the center and radius of F₂
def F₂_center : ℝ × ℝ := (3, 0)
def F₂_radius : ℝ := 1

-- Define the trajectory of the center of circle P
def trajectory (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1

-- Theorem statement
theorem circle_trajectory :
  ∀ (x y r : ℝ),
  (∃ (x₁ y₁ : ℝ), F₁ x₁ y₁ ∧ (x - x₁)^2 + (y - y₁)^2 = r^2) →
  (∃ (x₂ y₂ : ℝ), F₂ x₂ y₂ ∧ (x - x₂)^2 + (y - y₂)^2 = r^2) →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_circle_trajectory_l485_48592


namespace NUMINAMATH_CALUDE_money_distribution_l485_48579

theorem money_distribution (x y z : ℝ) : 
  x + (y/2 + z/2) = 90 →
  y + (x/2 + z/2) = 70 →
  z + (x/2 + y/2) = 56 →
  y = 32 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l485_48579


namespace NUMINAMATH_CALUDE_min_abs_E_is_zero_l485_48519

/-- Given a real-valued function E, prove that its minimum absolute value is 0
    when the minimum of |E(x)| + |x + 6| + |x - 5| is 11 for all real x. -/
theorem min_abs_E_is_zero (E : ℝ → ℝ) : 
  (∀ x, |E x| + |x + 6| + |x - 5| ≥ 11) → 
  (∃ x, |E x| + |x + 6| + |x - 5| = 11) → 
  ∃ x, |E x| = 0 :=
sorry

end NUMINAMATH_CALUDE_min_abs_E_is_zero_l485_48519


namespace NUMINAMATH_CALUDE_exponential_continuous_l485_48594

/-- The exponential function is continuous for any positive base -/
theorem exponential_continuous (a : ℝ) (h : a > 0) :
  Continuous (fun x => a^x) :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_continuous_l485_48594


namespace NUMINAMATH_CALUDE_triangular_pyramid_projections_not_equal_l485_48558

/-- Represents a three-dimensional solid object -/
structure Solid :=
  (shape : Type)

/-- Represents an orthogonal projection (view) of a solid -/
structure Projection :=
  (shape : Type)
  (size : ℝ)

/-- Returns the front projection of a solid -/
def front_view (s : Solid) : Projection :=
  sorry

/-- Returns the top projection of a solid -/
def top_view (s : Solid) : Projection :=
  sorry

/-- Returns the side projection of a solid -/
def side_view (s : Solid) : Projection :=
  sorry

/-- Defines a triangular pyramid -/
def triangular_pyramid : Solid :=
  sorry

/-- Theorem stating that a triangular pyramid cannot have all three
    orthogonal projections of the same shape and size -/
theorem triangular_pyramid_projections_not_equal :
  ∃ (p1 p2 : Projection), 
    (p1 = front_view triangular_pyramid ∧
     p2 = top_view triangular_pyramid ∧
     p1 ≠ p2) ∨
    (p1 = front_view triangular_pyramid ∧
     p2 = side_view triangular_pyramid ∧
     p1 ≠ p2) ∨
    (p1 = top_view triangular_pyramid ∧
     p2 = side_view triangular_pyramid ∧
     p1 ≠ p2) :=
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_projections_not_equal_l485_48558


namespace NUMINAMATH_CALUDE_sequence_a_500th_term_l485_48591

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1007 ∧ 
  a 2 = 1008 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem sequence_a_500th_term (a : ℕ → ℕ) (h : sequence_a a) : a 500 = 1173 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_500th_term_l485_48591


namespace NUMINAMATH_CALUDE_complex_modulus_l485_48588

theorem complex_modulus (z : ℂ) (h : z * (2 + Complex.I) = Complex.I ^ 10) :
  Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l485_48588


namespace NUMINAMATH_CALUDE_big_suv_to_normal_car_ratio_l485_48536

/-- Represents the time in minutes for each task when washing a normal car -/
structure NormalCarWashTime where
  windows : Nat
  body : Nat
  tires : Nat
  waxing : Nat

/-- Calculates the total time to wash a normal car -/
def normalCarTotalTime (t : NormalCarWashTime) : Nat :=
  t.windows + t.body + t.tires + t.waxing

/-- Represents the washing scenario -/
structure CarWashScenario where
  normalCarTime : NormalCarWashTime
  normalCarCount : Nat
  totalTime : Nat

/-- Theorem: The ratio of time taken to wash the big SUV to the time taken to wash a normal car is 2:1 -/
theorem big_suv_to_normal_car_ratio 
  (scenario : CarWashScenario) 
  (h1 : scenario.normalCarTime = ⟨4, 7, 4, 9⟩) 
  (h2 : scenario.normalCarCount = 2) 
  (h3 : scenario.totalTime = 96) : 
  (scenario.totalTime - scenario.normalCarCount * normalCarTotalTime scenario.normalCarTime) / 
  (normalCarTotalTime scenario.normalCarTime) = 2 := by
  sorry


end NUMINAMATH_CALUDE_big_suv_to_normal_car_ratio_l485_48536


namespace NUMINAMATH_CALUDE_josie_shortage_l485_48585

def gift_amount : ℝ := 150
def cassette_count : ℕ := 5
def cassette_price : ℝ := 18
def headphone_count : ℕ := 2
def headphone_price : ℝ := 45
def vinyl_count : ℕ := 3
def vinyl_price : ℝ := 22
def magazine_count : ℕ := 4
def magazine_price : ℝ := 7

def total_cost : ℝ :=
  cassette_count * cassette_price +
  headphone_count * headphone_price +
  vinyl_count * vinyl_price +
  magazine_count * magazine_price

theorem josie_shortage : gift_amount - total_cost = -124 := by
  sorry

end NUMINAMATH_CALUDE_josie_shortage_l485_48585


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l485_48509

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ := n.factorial / k.factorial

theorem book_arrangement_proof :
  let total_books : ℕ := 7
  let identical_books : ℕ := 3
  let distinct_books : ℕ := 4
  let books_to_arrange : ℕ := total_books - 1
  number_of_arrangements books_to_arrange identical_books = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l485_48509


namespace NUMINAMATH_CALUDE_pet_snake_cost_l485_48512

def initial_amount : ℕ := 73
def amount_left : ℕ := 18

theorem pet_snake_cost : initial_amount - amount_left = 55 := by sorry

end NUMINAMATH_CALUDE_pet_snake_cost_l485_48512


namespace NUMINAMATH_CALUDE_gas_purchase_l485_48582

theorem gas_purchase (price_nc : ℝ) (amount : ℝ) : 
  price_nc > 0 →
  amount > 0 →
  price_nc * amount + (price_nc + 1) * amount = 50 →
  price_nc = 2 →
  amount = 10 := by
sorry

end NUMINAMATH_CALUDE_gas_purchase_l485_48582


namespace NUMINAMATH_CALUDE_third_term_of_sequence_l485_48532

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem third_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 21 = 12 →
  arithmetic_sequence a₁ d 22 = 15 →
  arithmetic_sequence a₁ d 3 = -42 :=
by sorry

end NUMINAMATH_CALUDE_third_term_of_sequence_l485_48532


namespace NUMINAMATH_CALUDE_parallelogram_area_l485_48508

/-- The area of a parallelogram with base 21 and height 11 is 231 -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 21 → 
    height = 11 → 
    area = base * height → 
    area = 231 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l485_48508


namespace NUMINAMATH_CALUDE_philip_banana_count_l485_48529

/-- The number of banana groups in Philip's collection -/
def banana_groups : ℕ := 2

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 145

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := banana_groups * bananas_per_group

theorem philip_banana_count : total_bananas = 290 := by
  sorry

end NUMINAMATH_CALUDE_philip_banana_count_l485_48529


namespace NUMINAMATH_CALUDE_correct_calculation_result_l485_48557

theorem correct_calculation_result : ∃ x : ℕ, (40 + x = 52) ∧ (20 * x = 240) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l485_48557


namespace NUMINAMATH_CALUDE_fraction_equality_implies_zero_l485_48540

theorem fraction_equality_implies_zero (x : ℝ) :
  (1 / (x - 1) = 2 / (x - 2)) ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_zero_l485_48540


namespace NUMINAMATH_CALUDE_cat_hunting_theorem_l485_48550

/-- The number of birds caught during the day -/
def day_birds : ℕ := 8

/-- The number of birds caught at night -/
def night_birds : ℕ := 2 * day_birds

/-- The total number of birds caught -/
def total_birds : ℕ := 24

theorem cat_hunting_theorem : 
  day_birds + night_birds = total_birds ∧ night_birds = 2 * day_birds :=
by sorry

end NUMINAMATH_CALUDE_cat_hunting_theorem_l485_48550


namespace NUMINAMATH_CALUDE_exists_double_application_negation_l485_48583

theorem exists_double_application_negation :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = -x := by
  sorry

end NUMINAMATH_CALUDE_exists_double_application_negation_l485_48583


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l485_48598

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 5474827
  let d := 12
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 ∧ k = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l485_48598


namespace NUMINAMATH_CALUDE_function_equation_solution_l485_48549

/-- Given a function f: ℝ → ℝ satisfying f(x-f(y)) = 1 - x - y for all x, y ∈ ℝ,
    prove that f(x) = 1/2 - x for all x ∈ ℝ. -/
theorem function_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x - f y) = 1 - x - y) : 
    ∀ x : ℝ, f x = 1/2 - x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l485_48549


namespace NUMINAMATH_CALUDE_intersection_point_l485_48593

/-- The system of linear equations representing two lines -/
def system (x y : ℝ) : Prop :=
  8 * x + 5 * y = 40 ∧ 3 * x - 10 * y = 15

/-- The theorem stating that (5, 0) is the unique solution to the system -/
theorem intersection_point : ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (5, 0) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l485_48593


namespace NUMINAMATH_CALUDE_mateo_deducted_salary_l485_48544

/-- Calculates the deducted salary for a worker given their weekly salary and number of absent days. -/
def deducted_salary (weekly_salary : ℚ) (absent_days : ℕ) : ℚ :=
  weekly_salary - (weekly_salary / 5 * absent_days)

/-- Proves that Mateo's deducted salary is correct given his weekly salary and absent days. -/
theorem mateo_deducted_salary :
  deducted_salary 791 4 = 158.2 := by
  sorry

end NUMINAMATH_CALUDE_mateo_deducted_salary_l485_48544


namespace NUMINAMATH_CALUDE_total_is_99_l485_48500

/-- The total number of ducks and ducklings in Mary's observation --/
def total_ducks_and_ducklings : ℕ := by
  -- Define the number of ducks in each group
  let ducks_group1 : ℕ := 2
  let ducks_group2 : ℕ := 6
  let ducks_group3 : ℕ := 9
  
  -- Define the number of ducklings per duck in each group
  let ducklings_per_duck_group1 : ℕ := 5
  let ducklings_per_duck_group2 : ℕ := 3
  let ducklings_per_duck_group3 : ℕ := 6
  
  -- Calculate the total number of ducks and ducklings
  exact ducks_group1 * ducklings_per_duck_group1 +
        ducks_group2 * ducklings_per_duck_group2 +
        ducks_group3 * ducklings_per_duck_group3 +
        ducks_group1 + ducks_group2 + ducks_group3

/-- Theorem stating that the total number of ducks and ducklings is 99 --/
theorem total_is_99 : total_ducks_and_ducklings = 99 := by
  sorry

end NUMINAMATH_CALUDE_total_is_99_l485_48500


namespace NUMINAMATH_CALUDE_cube_difference_l485_48530

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : 
  a^3 - b^3 = 124 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l485_48530


namespace NUMINAMATH_CALUDE_parenthesizations_of_triple_exponent_l485_48538

/-- Represents the number of distinct parenthesizations of 3^3^3^3 -/
def num_parenthesizations : ℕ := 5

/-- Represents the number of distinct values obtained from different parenthesizations of 3^3^3^3 -/
def num_distinct_values : ℕ := 5

/-- The expression 3^3^3^3 can be parenthesized in 5 different ways, resulting in 5 distinct values -/
theorem parenthesizations_of_triple_exponent :
  num_parenthesizations = num_distinct_values :=
by sorry

#check parenthesizations_of_triple_exponent

end NUMINAMATH_CALUDE_parenthesizations_of_triple_exponent_l485_48538


namespace NUMINAMATH_CALUDE_equation_solutions_l485_48522

theorem equation_solutions :
  ∀ x : ℝ, x ≥ 4 →
    ((x / (2 * Real.sqrt 2) + 5 * Real.sqrt 2 / 2) * Real.sqrt (x^3 - 64*x + 200) = x^2 + 6*x - 40) ↔
    (x = 6 ∨ x = Real.sqrt 13 + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l485_48522


namespace NUMINAMATH_CALUDE_henrys_classical_cds_l485_48572

/-- Given Henry's CD collection, prove the number of classical CDs --/
theorem henrys_classical_cds :
  ∀ (country rock classical : ℕ),
    country = 23 →
    country = rock + 3 →
    rock = 2 * classical →
    classical = 10 := by
  sorry

end NUMINAMATH_CALUDE_henrys_classical_cds_l485_48572


namespace NUMINAMATH_CALUDE_horner_rule_v₂_l485_48520

def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

def v₀ : ℝ := 4

def v₁ (x : ℝ) : ℝ := v₀ * x + 3

def v₂ (x : ℝ) : ℝ := v₁ x * x - 6

theorem horner_rule_v₂ : v₂ (-1) = -5 := by sorry

end NUMINAMATH_CALUDE_horner_rule_v₂_l485_48520
