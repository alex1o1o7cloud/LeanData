import Mathlib

namespace NUMINAMATH_GPT_length_PQ_eq_b_l1397_139799

open Real

variables {a b : ℝ} (h : a > b) (p : ℝ × ℝ) (h₁ : (p.fst / a) ^ 2 + (p.snd / b) ^ 2 = 1)
variables (F₁ F₂ : ℝ × ℝ) (P Q : ℝ × ℝ)
variable (Q_on_segment : Q.1 = (F₁.1 + F₂.1) / 2)
variable (equal_inradii : inradius (triangle P Q F₁) = inradius (triangle P Q F₂))

theorem length_PQ_eq_b : dist P Q = b :=
by
  sorry

end NUMINAMATH_GPT_length_PQ_eq_b_l1397_139799


namespace NUMINAMATH_GPT_part_I_part_II_l1397_139782

variables {x a : ℝ} (p : Prop) (q : Prop)

-- Proposition p
def prop_p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 ∧ a > 0

-- Proposition q
def prop_q (x : ℝ) : Prop := (x^2 - 2*x - 8 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Part (I)
theorem part_I (a : ℝ) (h : a = 1) : (prop_p x a) → (prop_q x) → (2 < x ∧ x < 4) :=
by
  sorry

-- Part (II)
theorem part_II (a : ℝ) : ¬(∃ x, prop_p x a) → ¬(∃ x, prop_q x) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1397_139782


namespace NUMINAMATH_GPT_part1_part2_l1397_139706

-- Condition: x = -1 is a solution to 2a + 4x = x + 5a
def is_solution_x (a x : ℤ) : Prop := 2 * a + 4 * x = x + 5 * a

-- Part 1: Prove a = -1 given x = -1
theorem part1 (x : ℤ) (h1 : x = -1) (h2 : is_solution_x a x) : a = -1 :=
by sorry

-- Condition: a = -1
def a_value (a : ℤ) : Prop := a = -1

-- Condition: ay + 6 = 6a + 2y
def equation_in_y (a y : ℤ) : Prop := a * y + 6 = 6 * a + 2 * y

-- Part 2: Prove y = 4 given a = -1
theorem part2 (a y : ℤ) (h1 : a_value a) (h2 : equation_in_y a y) : y = 4 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1397_139706


namespace NUMINAMATH_GPT_johnny_savings_l1397_139786

variable (S : ℤ) -- The savings in September.

theorem johnny_savings :
  (S + 49 + 46 - 58 = 67) → (S = 30) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_johnny_savings_l1397_139786


namespace NUMINAMATH_GPT_chord_length_l1397_139737

theorem chord_length (r : ℝ) (h : r = 15) :
  ∃ (cd : ℝ), cd = 13 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l1397_139737


namespace NUMINAMATH_GPT_unique_solution_a_exists_l1397_139716

open Real

noncomputable def equation (a x : ℝ) :=
  4 * a^2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x

theorem unique_solution_a_exists : 
  ∃! a : ℝ, ∃ x : ℝ, 0 < x ∧ equation a x :=
sorry

end NUMINAMATH_GPT_unique_solution_a_exists_l1397_139716


namespace NUMINAMATH_GPT_find_expression_for_f_l1397_139774

theorem find_expression_for_f (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2 + 6 * x) :
  ∀ x, f x = x^2 + 8 * x + 7 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_for_f_l1397_139774


namespace NUMINAMATH_GPT_ab_sum_l1397_139796

theorem ab_sum (A B C D : Nat) (h_digits: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_mult : A * (10 * C + D) = 1001 + 100 * A + 10 * B + A) : A + B = 1 := 
  sorry

end NUMINAMATH_GPT_ab_sum_l1397_139796


namespace NUMINAMATH_GPT_perfect_squares_perfect_square_plus_one_l1397_139747

theorem perfect_squares : (∃ n : ℕ, 2^n + 3 = (x : ℕ)^2) ↔ n = 0 ∨ n = 3 :=
by
  sorry

theorem perfect_square_plus_one : (∃ n : ℕ, 2^n + 1 = (x : ℕ)^2) ↔ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_perfect_squares_perfect_square_plus_one_l1397_139747


namespace NUMINAMATH_GPT_odd_number_divisibility_l1397_139781

theorem odd_number_divisibility (a : ℤ) (h : a % 2 = 1) : ∃ (k : ℤ), a^4 + 9 * (9 - 2 * a^2) = 16 * k :=
by
  sorry

end NUMINAMATH_GPT_odd_number_divisibility_l1397_139781


namespace NUMINAMATH_GPT_roots_separation_condition_l1397_139738

theorem roots_separation_condition (m n p q : ℝ)
  (h_1 : ∃ (x1 x2 : ℝ), x1 + x2 = -m ∧ x1 * x2 = n ∧ x1 ≠ x2)
  (h_2 : ∃ (x3 x4 : ℝ), x3 + x4 = -p ∧ x3 * x4 = q ∧ x3 ≠ x4)
  (h_3 : (∀ x1 x2 x3 x4 : ℝ, x1 + x2 = -m ∧ x1 * x2 = n ∧ x3 + x4 = -p ∧ x3 * x4 = q → 
         (x3 - x1) * (x3 - x2) * (x4 - x1) * (x4 - x2) < 0)) : 
  (n - q)^2 + (m - p) * (m * q - n * p) < 0 :=
sorry

end NUMINAMATH_GPT_roots_separation_condition_l1397_139738


namespace NUMINAMATH_GPT_num_partitions_of_staircase_l1397_139773

-- Definition of a staircase
def is_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ j → j ≤ i → i ≤ n → cells (i, j)

-- Number of partitions of a staircase of height n
def num_partitions (n : ℕ) : ℕ :=
  2^(n-1)

theorem num_partitions_of_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) :
  is_staircase n cells → (∃ p : ℕ, p = num_partitions n) :=
by
  intro h
  use (2^(n-1))
  sorry

end NUMINAMATH_GPT_num_partitions_of_staircase_l1397_139773


namespace NUMINAMATH_GPT_beetle_distance_l1397_139780

theorem beetle_distance :
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  dist1 + dist2 = 20 :=
by
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  show dist1 + dist2 = 20
  sorry

end NUMINAMATH_GPT_beetle_distance_l1397_139780


namespace NUMINAMATH_GPT_upstream_distance_l1397_139722

variable (Vb Vs Vdown Vup Dup : ℕ)

def boatInStillWater := Vb = 36
def speedStream := Vs = 12
def downstreamSpeed := Vdown = Vb + Vs
def upstreamSpeed := Vup = Vb - Vs
def timeEquality := 80 / Vdown = Dup / Vup

theorem upstream_distance (Vb Vs Vdown Vup Dup : ℕ) 
  (h1 : boatInStillWater Vb)
  (h2 : speedStream Vs)
  (h3 : downstreamSpeed Vb Vs Vdown)
  (h4 : upstreamSpeed Vb Vs Vup)
  (h5 : timeEquality Vdown Vup Dup) : Dup = 40 := 
sorry

end NUMINAMATH_GPT_upstream_distance_l1397_139722


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l1397_139718

theorem isosceles_right_triangle_area
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : c = a * Real.sqrt 2) 
  (area : ℝ) 
  (h_area : area = 50)
  (h3 : (1/2) * a * b = area) :
  (a + b + c) / area = 0.4 + 0.2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l1397_139718


namespace NUMINAMATH_GPT_expected_scurried_home_mn_sum_l1397_139719

theorem expected_scurried_home_mn_sum : 
  let expected_fraction : ℚ := (1/2 + 2/3 + 3/4 + 4/5 + 5/6 + 6/7 + 7/8)
  let m : ℕ := 37
  let n : ℕ := 7
  m + n = 44 := by
  sorry

end NUMINAMATH_GPT_expected_scurried_home_mn_sum_l1397_139719


namespace NUMINAMATH_GPT_roberto_outfits_l1397_139775

theorem roberto_outfits (trousers shirts jackets : ℕ) (restricted_shirt restricted_jacket : ℕ) 
  (h_trousers : trousers = 5) 
  (h_shirts : shirts = 6) 
  (h_jackets : jackets = 4) 
  (h_restricted_shirt : restricted_shirt = 1) 
  (h_restricted_jacket : restricted_jacket = 1) : 
  ((trousers * shirts * jackets) - (restricted_shirt * restricted_jacket * trousers) = 115) := 
  by 
    sorry

end NUMINAMATH_GPT_roberto_outfits_l1397_139775


namespace NUMINAMATH_GPT_number_of_distinct_intersections_l1397_139770

/-- The problem is to prove that the number of distinct intersection points
in the xy-plane for the graphs of the given equations is exactly 4. -/
theorem number_of_distinct_intersections :
  ∃ (S : Finset (ℝ × ℝ)), 
  (∀ p : ℝ × ℝ, p ∈ S ↔
    ((p.1 + p.2 = 7 ∨ 2 * p.1 - 3 * p.2 + 1 = 0) ∧
     (p.1 - p.2 - 2 = 0 ∨ 3 * p.1 + 2 * p.2 - 10 = 0))) ∧
  S.card = 4 :=
sorry

end NUMINAMATH_GPT_number_of_distinct_intersections_l1397_139770


namespace NUMINAMATH_GPT_samantha_last_name_length_l1397_139710

/-
Given:
1. Jamie’s last name "Grey" has 4 letters.
2. If Bobbie took 2 letters off her last name, her last name would have twice the length of Jamie’s last name.
3. Samantha’s last name has 3 fewer letters than Bobbie’s last name.

Prove:
- Samantha's last name contains 7 letters.
-/

theorem samantha_last_name_length : 
  ∀ (Jamie Bobbie Samantha : ℕ),
    Jamie = 4 →
    Bobbie - 2 = 2 * Jamie →
    Samantha = Bobbie - 3 →
    Samantha = 7 :=
by
  intros Jamie Bobbie Samantha hJamie hBobbie hSamantha
  sorry

end NUMINAMATH_GPT_samantha_last_name_length_l1397_139710


namespace NUMINAMATH_GPT_john_school_year_hours_l1397_139749

noncomputable def requiredHoursPerWeek (summerHoursPerWeek : ℕ) (summerWeeks : ℕ) 
                                       (summerEarnings : ℕ) (schoolWeeks : ℕ) 
                                       (schoolEarnings : ℕ) : ℕ :=
    schoolEarnings * summerHoursPerWeek * summerWeeks / (summerEarnings * schoolWeeks)

theorem john_school_year_hours :
  ∀ (summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings : ℕ),
    summerHoursPerWeek = 40 →
    summerWeeks = 10 →
    summerEarnings = 4000 →
    schoolWeeks = 50 →
    schoolEarnings = 4000 →
    requiredHoursPerWeek summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_john_school_year_hours_l1397_139749


namespace NUMINAMATH_GPT_raja_journey_distance_l1397_139754

theorem raja_journey_distance
  (T : ℝ) (D : ℝ)
  (H1 : T = 10)
  (H2 : ∀ t1 t2, t1 = D / 42 ∧ t2 = D / 48 → T = t1 + t2) :
  D = 224 :=
by
  sorry

end NUMINAMATH_GPT_raja_journey_distance_l1397_139754


namespace NUMINAMATH_GPT_total_area_of_squares_l1397_139700

theorem total_area_of_squares (x : ℝ) (hx : 4 * x^2 = 240) : 
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  2 * small_square_area + large_square_area = 360 :=
by
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  sorry

end NUMINAMATH_GPT_total_area_of_squares_l1397_139700


namespace NUMINAMATH_GPT_number_of_large_balls_l1397_139783

def smallBallRubberBands : ℕ := 50
def largeBallRubberBands : ℕ := 300
def totalRubberBands : ℕ := 5000
def smallBallsMade : ℕ := 22

def rubberBandsUsedForSmallBalls := smallBallsMade * smallBallRubberBands
def remainingRubberBands := totalRubberBands - rubberBandsUsedForSmallBalls

theorem number_of_large_balls :
  (remainingRubberBands / largeBallRubberBands) = 13 := by
  sorry

end NUMINAMATH_GPT_number_of_large_balls_l1397_139783


namespace NUMINAMATH_GPT_determine_a_l1397_139761

theorem determine_a
  (a b : ℝ)
  (P1 P2 : ℝ × ℝ)
  (direction_vector : ℝ × ℝ)
  (h1 : P1 = (-3, 4))
  (h2 : P2 = (4, -1))
  (h3 : direction_vector = (4 - (-3), -1 - 4))
  (h4 : b = a / 2)
  (h5 : direction_vector = (7, -5)) :
  a = -10 :=
sorry

end NUMINAMATH_GPT_determine_a_l1397_139761


namespace NUMINAMATH_GPT_smallest_positive_root_l1397_139735

noncomputable def alpha : ℝ := Real.arctan (2 / 9)
noncomputable def beta : ℝ := Real.arctan (6 / 7)

theorem smallest_positive_root :
  ∃ x > 0, (2 * Real.sin (6 * x) + 9 * Real.cos (6 * x) = 6 * Real.sin (2 * x) + 7 * Real.cos (2 * x))
    ∧ x = (alpha + beta) / 8 := sorry

end NUMINAMATH_GPT_smallest_positive_root_l1397_139735


namespace NUMINAMATH_GPT_range_of_a_l1397_139793

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, (2 * a - 1 ≤ x ∧ x ≤ a + 3)) →
  (-1 ≤ a ∧ a ≤ 0) :=
by
  -- Prove the theorem
  sorry

end NUMINAMATH_GPT_range_of_a_l1397_139793


namespace NUMINAMATH_GPT_remainder_of_A_div_by_9_l1397_139707

theorem remainder_of_A_div_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_of_A_div_by_9_l1397_139707


namespace NUMINAMATH_GPT_inequality_k_distance_comparison_l1397_139755

theorem inequality_k (k : ℝ) (x : ℝ) : 
  -3 < k ∧ k ≤ 0 → 2 * k * x^2 + k * x - 3/8 < 0 := sorry

theorem distance_comparison (a b : ℝ) (hab : a ≠ b) : 
  (abs ((a^2 + b^2) / 2 - (a + b)^2 / 4) > abs (a * b - (a + b)^2 / 4)) := sorry

end NUMINAMATH_GPT_inequality_k_distance_comparison_l1397_139755


namespace NUMINAMATH_GPT_smallest_integer_satisfies_inequality_l1397_139727

theorem smallest_integer_satisfies_inequality :
  ∃ (x : ℤ), (x^2 < 2 * x + 3) ∧ ∀ (y : ℤ), (y^2 < 2 * y + 3) → x ≤ y ∧ x = 0 :=
sorry

end NUMINAMATH_GPT_smallest_integer_satisfies_inequality_l1397_139727


namespace NUMINAMATH_GPT_determine_a_b_l1397_139703

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the first derivative of the function f
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Define the conditions given in the problem
def conditions (a b : ℝ) : Prop :=
  (f' 1 a b = 0) ∧ (f 1 a b = 10)

-- Provide the main theorem stating the required proof
theorem determine_a_b (a b : ℝ) (h : conditions a b) : a = 4 ∧ b = -11 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_a_b_l1397_139703


namespace NUMINAMATH_GPT_min_value_f_when_a_is_zero_inequality_holds_for_f_l1397_139712

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Problem (1): Prove the minimum value of f(x) when a = 0 is 2 - 2 * ln 2.
theorem min_value_f_when_a_is_zero : 
  (∃ x : ℝ, f x 0 = 2 - 2 * Real.log 2) :=
sorry

-- Problem (2): Prove that for a < (exp(1) / 2) - 1, f(x) > (exp(1) / 2) - 1 for all x in (0, +∞).
theorem inequality_holds_for_f :
  ∀ a : ℝ, a < (Real.exp 1) / 2 - 1 → 
  ∀ x : ℝ, 0 < x → f x a > (Real.exp 1) / 2 - 1 :=
sorry

end NUMINAMATH_GPT_min_value_f_when_a_is_zero_inequality_holds_for_f_l1397_139712


namespace NUMINAMATH_GPT_pairing_probability_l1397_139777

variable {students : Fin 28} (Alex Jamie : Fin 28)

theorem pairing_probability (h1 : ∀ (i j : Fin 28), i ≠ j) :
  ∃ p : ℚ, p = 1 / 27 ∧ 
  (∃ (A_J_pairs : Finset (Fin 28) × Finset (Fin 28)),
  A_J_pairs.1 = {Alex} ∧ A_J_pairs.2 = {Jamie}) -> p = 1 / 27
:= sorry

end NUMINAMATH_GPT_pairing_probability_l1397_139777


namespace NUMINAMATH_GPT_quadrant_classification_l1397_139769

theorem quadrant_classification :
  ∀ (x y : ℝ), (4 * x - 3 * y = 24) → (|x| = |y|) → 
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  intros x y h_line h_eqdist
  sorry

end NUMINAMATH_GPT_quadrant_classification_l1397_139769


namespace NUMINAMATH_GPT_stream_current_rate_l1397_139798

theorem stream_current_rate (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (3 * r + w) + 2 = 18 / (3 * r - w)) : w = 3 :=
  sorry

end NUMINAMATH_GPT_stream_current_rate_l1397_139798


namespace NUMINAMATH_GPT_problem_1_problem_2_l1397_139741

noncomputable def f (a b x : ℝ) := a * (x - 1)^2 + b * Real.log x

theorem problem_1 (a : ℝ) (h_deriv : ∀ x ≥ 2, (2 * a * x^2 - 2 * a * x + 1) / x ≤ 0) : 
  a ≤ -1 / 4 :=
sorry

theorem problem_2 (a : ℝ) (h_ineq : ∀ x ≥ 1, a * (x - 1)^2 + Real.log x ≤ x - 1) : 
  a ≤ 0 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1397_139741


namespace NUMINAMATH_GPT_tomatoes_picked_today_l1397_139771

theorem tomatoes_picked_today (initial yesterday_picked left_after_yesterday today_picked : ℕ)
  (h1 : initial = 160)
  (h2 : yesterday_picked = 56)
  (h3 : left_after_yesterday = 104)
  (h4 : initial - yesterday_picked = left_after_yesterday) :
  today_picked = 56 :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_picked_today_l1397_139771


namespace NUMINAMATH_GPT_num_large_posters_l1397_139762

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end NUMINAMATH_GPT_num_large_posters_l1397_139762


namespace NUMINAMATH_GPT_maximum_cards_without_equal_pair_sums_l1397_139752

def max_cards_no_equal_sum_pairs : ℕ :=
  let card_points := {x : ℕ | 1 ≤ x ∧ x ≤ 13}
  6

theorem maximum_cards_without_equal_pair_sums (deck : Finset ℕ) (h_deck : deck = {x : ℕ | 1 ≤ x ∧ x ≤ 13}) :
  ∃ S ⊆ deck, S.card = 6 ∧ ∀ {a b c d : ℕ}, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a + b = c + d → a = c ∧ b = d ∨ a = d ∧ b = c := 
sorry

end NUMINAMATH_GPT_maximum_cards_without_equal_pair_sums_l1397_139752


namespace NUMINAMATH_GPT_total_students_l1397_139779

-- Define the conditions
variables (S : ℕ) -- total number of students
variable (h1 : (3/5 : ℚ) * S + (1/5 : ℚ) * S + 10 = S)

-- State the theorem
theorem total_students (HS : S = 50) : 3 / 5 * S + 1 / 5 * S + 10 = S := by
  -- Here we declare the proof is to be filled in later.
  sorry

end NUMINAMATH_GPT_total_students_l1397_139779


namespace NUMINAMATH_GPT_h_at_8_l1397_139701

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

noncomputable def h (x : ℝ) : ℝ :=
  let a := 1
  let b := 1
  let c := 2
  (1/2) * (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_8 : h 8 = 147 := 
by 
  sorry

end NUMINAMATH_GPT_h_at_8_l1397_139701


namespace NUMINAMATH_GPT_fg_difference_l1397_139742

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 4 * x - 1

theorem fg_difference : f (g 3) - g (f 3) = -16 := by
  sorry

end NUMINAMATH_GPT_fg_difference_l1397_139742


namespace NUMINAMATH_GPT_shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l1397_139739

noncomputable def circle_C : Set (ℝ × ℝ) := { p | (p.1^2 + p.2^2 - 4*p.1 - 2*p.2) = 0 }

def point_M_in_circle : Prop :=
  (1, 0) ∈ circle_C

theorem shortest_chord_through_M_is_x_plus_y_minus_1_eq_0 :
  point_M_in_circle →
  ∃ (a b c : ℝ), a * 1 + b * 0 + c = 0 ∧
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x + y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l1397_139739


namespace NUMINAMATH_GPT_trig_identity_l1397_139713

theorem trig_identity (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 3 - Real.cos (2 * x)) : f (Real.cos x) = 3 + Real.cos (2 * x) :=
sorry

end NUMINAMATH_GPT_trig_identity_l1397_139713


namespace NUMINAMATH_GPT_ratio_tuesday_monday_l1397_139709

-- Define the conditions
variables (M T W : ℕ) (hM : M = 450) (hW : W = 300) (h_rel : W = T + 75)

-- Define the theorem
theorem ratio_tuesday_monday : (T : ℚ) / M = 1 / 2 :=
by
  -- Sorry means the proof has been omitted in Lean.
  sorry

end NUMINAMATH_GPT_ratio_tuesday_monday_l1397_139709


namespace NUMINAMATH_GPT_sector_area_l1397_139728

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) : 
    (1/2 * r^2 * θ) = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l1397_139728


namespace NUMINAMATH_GPT_solve_for_y_l1397_139748

theorem solve_for_y (x y : ℝ) (h₁ : x - y = 16) (h₂ : x + y = 4) : y = -6 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l1397_139748


namespace NUMINAMATH_GPT_solution_amount_of_solution_A_l1397_139723

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x + y = 140)
variables (h2 : 0.40 * x + 0.90 * y = 0.80 * 140)

-- State the theorem
theorem solution_amount_of_solution_A : x = 28 :=
by
  -- Here, the proof would be provided, but we replace it with sorry
  sorry

end NUMINAMATH_GPT_solution_amount_of_solution_A_l1397_139723


namespace NUMINAMATH_GPT_josanna_next_test_score_l1397_139745

theorem josanna_next_test_score :
  let scores := [75, 85, 65, 95, 70]
  let current_sum := scores.sum
  let current_average := current_sum / scores.length
  let desired_average := current_average + 10
  let new_test_count := scores.length + 1
  let desired_sum := desired_average * new_test_count
  let required_score := desired_sum - current_sum
  required_score = 138 :=
by
  sorry

end NUMINAMATH_GPT_josanna_next_test_score_l1397_139745


namespace NUMINAMATH_GPT_initial_pineapple_sweets_l1397_139736

-- Define constants for initial number of flavored sweets and actions taken
def initial_cherry_sweets : ℕ := 30
def initial_strawberry_sweets : ℕ := 40
def total_remaining_sweets : ℕ := 55

-- Define Aaron's actions
def aaron_eats_half_sweets (n : ℕ) : ℕ := n / 2
def aaron_gives_to_friend : ℕ := 5

-- Calculate remaining sweets after Aaron's actions
def remaining_cherry_sweets : ℕ := initial_cherry_sweets - (aaron_eats_half_sweets initial_cherry_sweets) - aaron_gives_to_friend
def remaining_strawberry_sweets : ℕ := initial_strawberry_sweets - (aaron_eats_half_sweets initial_strawberry_sweets)

-- Define the problem to prove
theorem initial_pineapple_sweets :
  (total_remaining_sweets - (remaining_cherry_sweets + remaining_strawberry_sweets)) * 2 = 50 :=
by sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_initial_pineapple_sweets_l1397_139736


namespace NUMINAMATH_GPT_find_a2_l1397_139756

variable (a : ℕ → ℝ) (d : ℝ)

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a n + d
axiom common_diff : d = 2
axiom geometric_mean : (a 4) ^ 2 = (a 5) * (a 2)

theorem find_a2 : a 2 = -8 := 
by 
  sorry

end NUMINAMATH_GPT_find_a2_l1397_139756


namespace NUMINAMATH_GPT_moles_of_NaOH_combined_l1397_139714

-- Given conditions
def moles_AgNO3 := 3
def moles_AgOH := 3
def balanced_ratio_AgNO3_NaOH := 1 -- 1:1 ratio as per the equation

-- Problem statement
theorem moles_of_NaOH_combined : 
  moles_AgOH = moles_AgNO3 → balanced_ratio_AgNO3_NaOH = 1 → 
  (∃ moles_NaOH, moles_NaOH = 3) := by
  sorry

end NUMINAMATH_GPT_moles_of_NaOH_combined_l1397_139714


namespace NUMINAMATH_GPT_files_per_folder_l1397_139702

-- Define the conditions
def initial_files : ℕ := 43
def deleted_files : ℕ := 31
def num_folders : ℕ := 2

-- Define the final problem statement
theorem files_per_folder :
  (initial_files - deleted_files) / num_folders = 6 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_files_per_folder_l1397_139702


namespace NUMINAMATH_GPT_youseff_blocks_from_office_l1397_139740

def blocks_to_office (x : ℕ) : Prop :=
  let walk_time := x  -- it takes x minutes to walk
  let bike_time := (20 * x) / 60  -- it takes (20 / 60) * x = (1 / 3) * x minutes to ride a bike
  walk_time = bike_time + 4  -- walking takes 4 more minutes than biking

theorem youseff_blocks_from_office (x : ℕ) (h : blocks_to_office x) : x = 6 :=
  sorry

end NUMINAMATH_GPT_youseff_blocks_from_office_l1397_139740


namespace NUMINAMATH_GPT_white_area_of_sign_l1397_139788

theorem white_area_of_sign : 
  let total_area := 6 * 18
  let F_area := 2 * (4 * 1) + 6 * 1
  let O_area := 2 * (6 * 1) + 2 * (4 * 1)
  let D_area := 6 * 1 + 4 * 1 + 4 * 1
  let total_black_area := F_area + O_area + O_area + D_area
  total_area - total_black_area = 40 :=
by
  sorry

end NUMINAMATH_GPT_white_area_of_sign_l1397_139788


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_correct_l1397_139767

noncomputable def isosceles_triangle_perimeter (x y : ℝ) : ℝ :=
  if x = y then 2 * x + y else if (2 * x > y ∧ y > 2 * x - y) ∨ (2 * y > x ∧ x > 2 * y - x) then 2 * y + x else 0

theorem isosceles_triangle_perimeter_correct (x y : ℝ) (h : |x - 5| + (y - 8)^2 = 0) :
  isosceles_triangle_perimeter x y = 18 ∨ isosceles_triangle_perimeter x y = 21 := by
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_correct_l1397_139767


namespace NUMINAMATH_GPT_smallest_positive_m_l1397_139720

theorem smallest_positive_m {m p q : ℤ} (h_eq : 12 * p^2 - m * p - 360 = 0) (h_pq : p * q = -30) :
  (m = 12 * (p + q)) → 0 < m → m = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_m_l1397_139720


namespace NUMINAMATH_GPT_exist_three_primes_sum_to_30_l1397_139785

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def less_than_twenty (n : ℕ) : Prop := n < 20

theorem exist_three_primes_sum_to_30 : 
  ∃ A B C : ℕ, is_prime A ∧ is_prime B ∧ is_prime C ∧ 
  less_than_twenty A ∧ less_than_twenty B ∧ less_than_twenty C ∧ 
  A + B + C = 30 :=
by 
  -- assume A = 2, prime and less than 20
  -- find B, C such that B and C are primes less than 20 and A + B + C = 30
  sorry

end NUMINAMATH_GPT_exist_three_primes_sum_to_30_l1397_139785


namespace NUMINAMATH_GPT_correct_operation_is_multiplication_by_3_l1397_139797

theorem correct_operation_is_multiplication_by_3
  (x : ℝ)
  (percentage_error : ℝ)
  (correct_result : ℝ := 3 * x)
  (incorrect_result : ℝ := x / 5)
  (error_percentage : ℝ := (correct_result - incorrect_result) / correct_result * 100) :
  percentage_error = 93.33333333333333 → correct_result / x = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_operation_is_multiplication_by_3_l1397_139797


namespace NUMINAMATH_GPT_common_difference_zero_l1397_139759

theorem common_difference_zero (a b c : ℕ) 
  (h_seq : ∃ d : ℕ, a = b + d ∧ b = c + d)
  (h_eq : (c - b) / a + (a - c) / b + (b - a) / c = 0) : 
  ∀ d : ℕ, d = 0 :=
by sorry

end NUMINAMATH_GPT_common_difference_zero_l1397_139759


namespace NUMINAMATH_GPT_min_time_to_cook_cakes_l1397_139733

theorem min_time_to_cook_cakes (cakes : ℕ) (pot_capacity : ℕ) (time_per_side : ℕ) 
  (h1 : cakes = 3) (h2 : pot_capacity = 2) (h3 : time_per_side = 5) : 
  ∃ t, t = 15 := by
  sorry

end NUMINAMATH_GPT_min_time_to_cook_cakes_l1397_139733


namespace NUMINAMATH_GPT_ball_arrangements_l1397_139726

-- Define the structure of the boxes and balls
structure BallDistributions where
  white_balls_box1 : ℕ
  black_balls_box1 : ℕ
  white_balls_box2 : ℕ
  black_balls_box2 : ℕ
  white_balls_box3 : ℕ
  black_balls_box3 : ℕ

-- Problem conditions
def valid_distribution (d : BallDistributions) : Prop :=
  d.white_balls_box1 + d.black_balls_box1 ≥ 2 ∧
  d.white_balls_box2 + d.black_balls_box2 ≥ 2 ∧
  d.white_balls_box3 + d.black_balls_box3 ≥ 2 ∧
  d.white_balls_box1 ≥ 1 ∧
  d.black_balls_box1 ≥ 1 ∧
  d.white_balls_box2 ≥ 1 ∧
  d.black_balls_box2 ≥ 1 ∧
  d.white_balls_box3 ≥ 1 ∧
  d.black_balls_box3 ≥ 1

def total_white_balls (d : BallDistributions) : ℕ :=
  d.white_balls_box1 + d.white_balls_box2 + d.white_balls_box3

def total_black_balls (d : BallDistributions) : ℕ :=
  d.black_balls_box1 + d.black_balls_box2 + d.black_balls_box3

def correct_distribution (d : BallDistributions) : Prop :=
  total_white_balls d = 4 ∧ total_black_balls d = 5

-- Main theorem to prove
theorem ball_arrangements : ∃ (d : BallDistributions), valid_distribution d ∧ correct_distribution d ∧ (number_of_distributions = 18) :=
  sorry

end NUMINAMATH_GPT_ball_arrangements_l1397_139726


namespace NUMINAMATH_GPT_choir_members_count_l1397_139750

theorem choir_members_count : ∃ n : ℕ, n = 226 ∧ 
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (200 < n ∧ n < 300) :=
by
  sorry

end NUMINAMATH_GPT_choir_members_count_l1397_139750


namespace NUMINAMATH_GPT_part_a_part_b_l1397_139743

-- Part (a)

theorem part_a : ∃ (a b : ℕ), 2015^2 + 2017^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

-- Part (b)

theorem part_b (k n : ℕ) : ∃ (a b : ℕ), (2 * k + 1)^2 + (2 * n + 1)^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1397_139743


namespace NUMINAMATH_GPT_trigonometric_identity_l1397_139751

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1397_139751


namespace NUMINAMATH_GPT_ensure_A_win_product_l1397_139725

theorem ensure_A_win_product {s : Finset ℕ} (h1 : s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h2 : 8 ∈ s) (h3 : 5 ∈ s) :
  (4 ∈ s ∧ 6 ∈ s ∧ 7 ∈ s) →
  4 * 6 * 7 = 168 := 
by 
  intro _ 
  exact Nat.mul_assoc 4 6 7

end NUMINAMATH_GPT_ensure_A_win_product_l1397_139725


namespace NUMINAMATH_GPT_each_niece_gets_13_l1397_139789

-- Define the conditions
def total_sandwiches : ℕ := 143
def number_of_nieces : ℕ := 11

-- Prove that each niece can get 13 ice cream sandwiches
theorem each_niece_gets_13 : total_sandwiches / number_of_nieces = 13 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_each_niece_gets_13_l1397_139789


namespace NUMINAMATH_GPT_inequalities_indeterminate_l1397_139717

variable (s x y z : ℝ)

theorem inequalities_indeterminate (h_s : s > 0) (h_ineq : s * x > z * y) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (¬ (x > z)) ∨ (¬ (-x > -z)) ∨ (¬ (s > z / x)) ∨ (¬ (s < y / x)) :=
by sorry

end NUMINAMATH_GPT_inequalities_indeterminate_l1397_139717


namespace NUMINAMATH_GPT_solve_for_x_l1397_139790

theorem solve_for_x (x : ℝ) (h : (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4)) : x = 257 / 16 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1397_139790


namespace NUMINAMATH_GPT_parts_of_milk_in_drink_A_l1397_139734

theorem parts_of_milk_in_drink_A (x : ℝ) (h : 63 * (4 * x) / (7 * (x + 3)) = 63 * 3 / (x + 3) + 21) : x = 16.8 :=
by
  sorry

end NUMINAMATH_GPT_parts_of_milk_in_drink_A_l1397_139734


namespace NUMINAMATH_GPT_rachel_age_when_emily_half_age_l1397_139704

theorem rachel_age_when_emily_half_age 
  (E_0 : ℕ) (R_0 : ℕ) (h1 : E_0 = 20) (h2 : R_0 = 24) 
  (age_diff : R_0 - E_0 = 4) : 
  ∃ R : ℕ, ∃ E : ℕ, E = R / 2 ∧ R = E + 4 ∧ R = 8 :=
by
  sorry

end NUMINAMATH_GPT_rachel_age_when_emily_half_age_l1397_139704


namespace NUMINAMATH_GPT_probability_correct_l1397_139753

def elenaNameLength : Nat := 5
def markNameLength : Nat := 4
def juliaNameLength : Nat := 5
def totalCards : Nat := elenaNameLength + markNameLength + juliaNameLength

-- Without replacement, drawing three cards from 14 cards randomly
def probabilityThreeDifferentSources : ℚ := 
  (elenaNameLength / totalCards) * (markNameLength / (totalCards - 1)) * (juliaNameLength / (totalCards - 2))

def totalPermutations : Nat := 6  -- EMJ, EJM, MEJ, MJE, JEM, JME

def requiredProbability : ℚ := totalPermutations * probabilityThreeDifferentSources

theorem probability_correct :
  requiredProbability = 25 / 91 := by
  sorry

end NUMINAMATH_GPT_probability_correct_l1397_139753


namespace NUMINAMATH_GPT_imaginary_part_of_complex_l1397_139758

theorem imaginary_part_of_complex : ∀ z : ℂ, z = i^2 * (1 + i) → z.im = -1 :=
by
  intro z
  intro h
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_l1397_139758


namespace NUMINAMATH_GPT_train_length_l1397_139730

theorem train_length
  (time : ℝ) (man_speed train_speed : ℝ) (same_direction : Prop)
  (h_time : time = 62.99496040316775)
  (h_man_speed : man_speed = 6)
  (h_train_speed : train_speed = 30)
  (h_same_direction : same_direction) :
  (train_speed - man_speed) * (1000 / 3600) * time = 1259.899208063355 := 
sorry

end NUMINAMATH_GPT_train_length_l1397_139730


namespace NUMINAMATH_GPT_BRAIN_7225_cycle_line_number_l1397_139757

def BRAIN_cycle : Nat := 5
def _7225_cycle : Nat := 4

theorem BRAIN_7225_cycle_line_number : Nat.lcm BRAIN_cycle _7225_cycle = 20 :=
by
  sorry

end NUMINAMATH_GPT_BRAIN_7225_cycle_line_number_l1397_139757


namespace NUMINAMATH_GPT_division_of_fractions_l1397_139732

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end NUMINAMATH_GPT_division_of_fractions_l1397_139732


namespace NUMINAMATH_GPT_no_real_solutions_l1397_139731

theorem no_real_solutions (x : ℝ) : (x - 3 * x + 7)^2 + 1 ≠ -|x| :=
by
  -- The statement of the theorem is sufficient; the proof is not needed as per indicated instructions.
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1397_139731


namespace NUMINAMATH_GPT_mutually_exclusive_event_3_l1397_139715

-- Definitions based on the conditions.
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Events based on problem conditions
def event_1 (a b : ℕ) : Prop := is_even a ∧ is_odd b ∨ is_odd a ∧ is_even b
def event_2 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_odd a ∧ is_odd b
def event_3 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_even a ∧ is_even b
def event_4 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ (is_even a ∨ is_even b)

-- Problem: Proving that event_3 is mutually exclusive with other events.
theorem mutually_exclusive_event_3 :
  ∀ (a b : ℕ), (event_3 a b) → ¬ (event_1 a b ∨ event_2 a b ∨ event_4 a b) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_event_3_l1397_139715


namespace NUMINAMATH_GPT_jimmy_change_l1397_139768

def cost_of_pens (num_pens : ℕ) (cost_per_pen : ℕ): ℕ := num_pens * cost_per_pen
def cost_of_notebooks (num_notebooks : ℕ) (cost_per_notebook : ℕ): ℕ := num_notebooks * cost_per_notebook
def cost_of_folders (num_folders : ℕ) (cost_per_folder : ℕ): ℕ := num_folders * cost_per_folder

def total_cost : ℕ :=
  cost_of_pens 3 1 + cost_of_notebooks 4 3 + cost_of_folders 2 5

def paid_amount : ℕ := 50

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end NUMINAMATH_GPT_jimmy_change_l1397_139768


namespace NUMINAMATH_GPT_find_scalars_l1397_139721

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1],
    ![4, 3]]

def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0],
    ![0, 1]]

theorem find_scalars (r s : ℤ) (h : B^6 = r • B + s • I) :
  (r = 1125) ∧ (s = -1875) :=
sorry

end NUMINAMATH_GPT_find_scalars_l1397_139721


namespace NUMINAMATH_GPT_find_slope_of_parallel_line_l1397_139744

-- Define the condition that line1 is parallel to line2.
def lines_parallel (k : ℝ) : Prop :=
  k = -3

-- The theorem that proves the condition given.
theorem find_slope_of_parallel_line (k : ℝ) (h : lines_parallel k) : k = -3 :=
by
  exact h

end NUMINAMATH_GPT_find_slope_of_parallel_line_l1397_139744


namespace NUMINAMATH_GPT_sum_of_numbers_Carolyn_removes_l1397_139705

noncomputable def game_carolyn_paul_sum : ℕ :=
  let initial_list := [1, 2, 3, 4, 5]
  let removed_by_paul := [3, 4]
  let removed_by_carolyn := [1, 2, 5]
  removed_by_carolyn.sum

theorem sum_of_numbers_Carolyn_removes :
  game_carolyn_paul_sum = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_Carolyn_removes_l1397_139705


namespace NUMINAMATH_GPT_train_passing_time_l1397_139791

theorem train_passing_time 
  (length_of_train : ℕ) 
  (length_of_platform : ℕ) 
  (time_to_pass_pole : ℕ) 
  (speed_of_train : ℕ) 
  (combined_length : ℕ) 
  (time_to_pass_platform : ℕ) 
  (h1 : length_of_train = 240) 
  (h2 : length_of_platform = 650)
  (h3 : time_to_pass_pole = 24)
  (h4 : speed_of_train = length_of_train / time_to_pass_pole)
  (h5 : combined_length = length_of_train + length_of_platform)
  (h6 : time_to_pass_platform = combined_length / speed_of_train) : 
  time_to_pass_platform = 89 :=
sorry

end NUMINAMATH_GPT_train_passing_time_l1397_139791


namespace NUMINAMATH_GPT_real_coefficient_polynomials_with_special_roots_l1397_139711

noncomputable def P1 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) * (Polynomial.X ^ 2 - Polynomial.X + 1)
noncomputable def P2 : Polynomial ℝ := (Polynomial.X + 1) ^ 3 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2)
noncomputable def P3 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 3 * (Polynomial.X - 2)
noncomputable def P4 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 3
noncomputable def P5 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2)
noncomputable def P6 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2) ^ 2
noncomputable def P7 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 2

theorem real_coefficient_polynomials_with_special_roots (P : Polynomial ℝ) :
  (∀ α, Polynomial.IsRoot P α → Polynomial.IsRoot P (1 - α) ∧ Polynomial.IsRoot P (1 / α)) →
  P = P1 ∨ P = P2 ∨ P = P3 ∨ P = P4 ∨ P = P5 ∨ P = P6 ∨ P = P7 :=
  sorry

end NUMINAMATH_GPT_real_coefficient_polynomials_with_special_roots_l1397_139711


namespace NUMINAMATH_GPT_sum_ge_3_implies_one_ge_2_l1397_139772

theorem sum_ge_3_implies_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_ge_3_implies_one_ge_2_l1397_139772


namespace NUMINAMATH_GPT_total_wheels_in_garage_l1397_139794

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end NUMINAMATH_GPT_total_wheels_in_garage_l1397_139794


namespace NUMINAMATH_GPT_bowling_average_decrease_l1397_139778

theorem bowling_average_decrease
    (initial_average : ℝ) (wickets_last_match : ℝ) (runs_last_match : ℝ)
    (average_decrease : ℝ) (W : ℝ)
    (H_initial : initial_average = 12.4)
    (H_wickets_last_match : wickets_last_match = 6)
    (H_runs_last_match : runs_last_match = 26)
    (H_average_decrease : average_decrease = 0.4) :
    W = 115 :=
by
  sorry

end NUMINAMATH_GPT_bowling_average_decrease_l1397_139778


namespace NUMINAMATH_GPT_simplify_fraction_l1397_139763

theorem simplify_fraction : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_l1397_139763


namespace NUMINAMATH_GPT_triangle_area_is_24_l1397_139787

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (0, 6)
def C : point := (8, 10)

def triangle_area (A B C : point) : ℝ := 
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_24 : triangle_area A B C = 24 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_triangle_area_is_24_l1397_139787


namespace NUMINAMATH_GPT_not_both_perfect_squares_l1397_139795

theorem not_both_perfect_squares (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ a b : ℕ, (n+1) * 2^n = a^2 ∧ (n+3) * 2^(n + 2) = b^2) :=
sorry

end NUMINAMATH_GPT_not_both_perfect_squares_l1397_139795


namespace NUMINAMATH_GPT_product_of_coefficients_l1397_139708

theorem product_of_coefficients (b c : ℤ)
  (H1 : ∀ r, r^2 - 2 * r - 1 = 0 → r^5 - b * r - c = 0):
  b * c = 348 :=
by
  -- Solution steps would go here
  sorry

end NUMINAMATH_GPT_product_of_coefficients_l1397_139708


namespace NUMINAMATH_GPT_total_rainfall_in_2004_l1397_139784

noncomputable def average_monthly_rainfall_2003 : ℝ := 35.0
noncomputable def average_monthly_rainfall_2004 : ℝ := average_monthly_rainfall_2003 + 4.0
noncomputable def total_rainfall_2004 : ℝ := 
  let regular_months := 11 * average_monthly_rainfall_2004
  let daily_rainfall_feb := average_monthly_rainfall_2004 / 30
  let feb_rain := daily_rainfall_feb * 29 
  regular_months + feb_rain

theorem total_rainfall_in_2004 : total_rainfall_2004 = 466.7 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_in_2004_l1397_139784


namespace NUMINAMATH_GPT_ellen_smoothie_l1397_139776

theorem ellen_smoothie :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries_used := total_ingredients - (yogurt + orange_juice)
  strawberries_used = 0.2 := by
  sorry

end NUMINAMATH_GPT_ellen_smoothie_l1397_139776


namespace NUMINAMATH_GPT_cello_viola_pairs_are_70_l1397_139764

-- Given conditions
def cellos : ℕ := 800
def violas : ℕ := 600
def pair_probability : ℝ := 0.00014583333333333335

-- Theorem statement translating the mathematical problem
theorem cello_viola_pairs_are_70 (n : ℕ) (h1 : cellos = 800) (h2 : violas = 600) (h3 : pair_probability = 0.00014583333333333335) :
  n = 70 :=
sorry

end NUMINAMATH_GPT_cello_viola_pairs_are_70_l1397_139764


namespace NUMINAMATH_GPT_limit_expression_l1397_139765

theorem limit_expression :
  (∀ (n : ℕ), ∃ l : ℝ, 
    ∀ ε > 0, ∃ N : ℕ, n > N → 
      abs (( (↑(n) + 1)^3 - (↑(n) - 1)^3) / ((↑(n) + 1)^2 + (↑(n) - 1)^2) - l) < ε) 
  → l = 3 :=
sorry

end NUMINAMATH_GPT_limit_expression_l1397_139765


namespace NUMINAMATH_GPT_part1_part2_l1397_139792

theorem part1 (u v w : ℤ) (h_uv : gcd u v = 1) (h_vw : gcd v w = 1) (h_wu : gcd w u = 1) 
: gcd (u * v + v * w + w * u) (u * v * w) = 1 :=
sorry

theorem part2 (u v w : ℤ) (b := u * v + v * w + w * u) (c := u * v * w) (h : gcd b c = 1) 
: gcd u v = 1 ∧ gcd v w = 1 ∧ gcd w u = 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1397_139792


namespace NUMINAMATH_GPT_meal_cost_l1397_139766

theorem meal_cost (M : ℝ) (h1 : 3 * M + 15 = 45) : M = 10 :=
by
  sorry

end NUMINAMATH_GPT_meal_cost_l1397_139766


namespace NUMINAMATH_GPT_min_people_liking_both_l1397_139760

theorem min_people_liking_both (total : ℕ) (Beethoven : ℕ) (Chopin : ℕ) 
    (total_eq : total = 150) (Beethoven_eq : Beethoven = 120) (Chopin_eq : Chopin = 95) : 
    ∃ (both : ℕ), both = 65 := 
by 
  have H := Beethoven + Chopin - total
  sorry

end NUMINAMATH_GPT_min_people_liking_both_l1397_139760


namespace NUMINAMATH_GPT_lost_card_number_l1397_139729

theorem lost_card_number (p : ℕ) (c : ℕ) (h : 0 ≤ c ∧ c ≤ 9)
  (sum_remaining_cards : 10 * p + 45 - (p + c) = 2012) : p + c = 223 := by
  sorry

end NUMINAMATH_GPT_lost_card_number_l1397_139729


namespace NUMINAMATH_GPT_find_y_l1397_139724

theorem find_y (t : ℝ) (x : ℝ := 3 - 2 * t) (y : ℝ := 5 * t + 6) (h : x = 1) : y = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1397_139724


namespace NUMINAMATH_GPT_train_length_l1397_139746

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℕ) 
  (h1 : speed_kmh = 180)
  (h2 : time_s = 18)
  (h3 : 1 = 1000 / 3600) :
  length_m = (speed_kmh * 1000 / 3600) * time_s :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1397_139746
