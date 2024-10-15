import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_2015_powers_l3745_374585

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that a number ends with 5 -/
def ends_with_5 (n : ℕ) : Prop := units_digit n = 5

/-- The property that powers of numbers ending in 5 always end in 5 for exponents ≥ 1 -/
def power_ends_with_5 (n : ℕ) : Prop := 
  ends_with_5 n → ∀ k : ℕ, k ≥ 1 → ends_with_5 (n^k)

theorem units_digit_of_2015_powers : 
  ends_with_5 2015 → 
  power_ends_with_5 2015 → 
  units_digit (2015^2 + 2015^0 + 2015^1 + 2015^5) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_2015_powers_l3745_374585


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3745_374560

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2*x + 1/(3*y)) * (2*x + 1/(3*y) - 2023) + (3*y + 1/(2*x)) * (3*y + 1/(2*x) - 2023) ≥ -2050529.5 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
  (2*x + 1/(3*y)) * (2*x + 1/(3*y) - 2023) + (3*y + 1/(2*x)) * (3*y + 1/(2*x) - 2023) = -2050529.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3745_374560


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l3745_374573

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ
  longer_base_length : longer_base = 117
  midpoint_segment_length : midpoint_segment = 5
  midpoint_segment_property : midpoint_segment = (longer_base - shorter_base) / 2

/-- Theorem stating that the shorter base of the trapezoid is 107 -/
theorem trapezoid_shorter_base (t : Trapezoid) : t.shorter_base = 107 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l3745_374573


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l3745_374556

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x - 5

-- State the theorem
theorem monotonic_increasing_condition (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → m ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l3745_374556


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_implies_a_range_l3745_374516

/-- A function f is monotonic on an interval [a, b] if it is either
    nondecreasing or nonincreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

theorem quadratic_monotonicity_implies_a_range (a : ℝ) :
  IsMonotonic (fun x => x^2 - 2*a*x - 3) 1 2 → a ≤ 1 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_implies_a_range_l3745_374516


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_hexagonal_pyramid_l3745_374588

/-- The radius of a sphere circumscribed around a regular hexagonal pyramid -/
theorem circumscribed_sphere_radius_hexagonal_pyramid 
  (a b : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < b) 
  (h₃ : a < b) : 
  ∃ R : ℝ, R = b^2 / (2 * Real.sqrt (b^2 - a^2)) ∧ 
  R > 0 ∧
  R * 2 * Real.sqrt (b^2 - a^2) = b^2 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_hexagonal_pyramid_l3745_374588


namespace NUMINAMATH_CALUDE_symmetry_center_of_f_l3745_374504

/-- Given a function f(x) and a constant θ, prove that (0,0) is one of the symmetry centers of the graph of f(x). -/
theorem symmetry_center_of_f (θ : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * Real.cos (2 * x + θ) * Real.sin θ - Real.sin (2 * (x + θ))
  (0, 0) ∈ {p : ℝ × ℝ | ∀ x, f (p.1 + x) = f (p.1 - x)} :=
by sorry

end NUMINAMATH_CALUDE_symmetry_center_of_f_l3745_374504


namespace NUMINAMATH_CALUDE_alicia_remaining_masks_l3745_374514

/-- The number of sets of masks Alicia had initially -/
def initial_sets : ℕ := 90

/-- The number of sets of masks Alicia gave away -/
def given_away : ℕ := 51

/-- The number of sets of masks left in Alicia's collection -/
def remaining_sets : ℕ := initial_sets - given_away

theorem alicia_remaining_masks : remaining_sets = 39 := by
  sorry

end NUMINAMATH_CALUDE_alicia_remaining_masks_l3745_374514


namespace NUMINAMATH_CALUDE_max_m_value_l3745_374535

/-- A point in the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Definition of a valid configuration -/
def ValidConfig (n : ℕ) (m : ℕ) (points : Fin (m + 2) → Point) : Prop :=
  (n % 2 = 1) ∧ 
  (points 0 = ⟨0, 1⟩) ∧ 
  (points (Fin.last m) = ⟨n + 1, n⟩) ∧ 
  (∀ i : Fin m, 1 ≤ (points i.succ).x ∧ (points i.succ).x ≤ n ∧ 
                1 ≤ (points i.succ).y ∧ (points i.succ).y ≤ n) ∧
  (∀ i : Fin (m + 1), i.val % 2 = 0 → (points i).y = (points i.succ).y) ∧
  (∀ i : Fin (m + 1), i.val % 2 = 1 → (points i).x = (points i.succ).x) ∧
  (∀ i j : Fin (m + 1), i < j → 
    ((points i).x = (points i.succ).x ∧ (points j).x = (points j.succ).x → 
      (points i).x ≠ (points j).x) ∨
    ((points i).y = (points i.succ).y ∧ (points j).y = (points j.succ).y → 
      (points i).y ≠ (points j).y))

/-- The main theorem -/
theorem max_m_value (n : ℕ) : 
  (n % 2 = 1) → (∃ m : ℕ, ∃ points : Fin (m + 2) → Point, ValidConfig n m points) → 
  (∀ k : ℕ, ∀ points : Fin (k + 2) → Point, ValidConfig n k points → k ≤ n * (n - 1)) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l3745_374535


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3745_374530

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3745_374530


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3745_374526

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3745_374526


namespace NUMINAMATH_CALUDE_num_cows_is_24_l3745_374542

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := sorry

/-- The total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- The total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- Theorem stating that the number of cows is 24 given the conditions -/
theorem num_cows_is_24 : 
  (total_legs = 2 * total_heads + 48) → num_cows = 24 := by
  sorry

end NUMINAMATH_CALUDE_num_cows_is_24_l3745_374542


namespace NUMINAMATH_CALUDE_particle_paths_l3745_374555

theorem particle_paths (n k : ℕ) : 
  (n = 5 ∧ k = 3) → (Nat.choose n ((n + k) / 2) = 5) ∧
  (n = 20 ∧ k = 16) → (Nat.choose n ((n + k) / 2) = 190) :=
by sorry

end NUMINAMATH_CALUDE_particle_paths_l3745_374555


namespace NUMINAMATH_CALUDE_sum_1_to_140_mod_7_l3745_374593

theorem sum_1_to_140_mod_7 : 
  (List.range 140).sum % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_1_to_140_mod_7_l3745_374593


namespace NUMINAMATH_CALUDE_digits_divisible_by_3_in_base_4_of_375_l3745_374518

def base_4_representation (n : ℕ) : List ℕ :=
  sorry

def count_divisible_by_3 (digits : List ℕ) : ℕ :=
  sorry

theorem digits_divisible_by_3_in_base_4_of_375 :
  count_divisible_by_3 (base_4_representation 375) = 2 :=
sorry

end NUMINAMATH_CALUDE_digits_divisible_by_3_in_base_4_of_375_l3745_374518


namespace NUMINAMATH_CALUDE_prize_probability_after_addition_l3745_374537

/-- Given a box with prizes, this function calculates the probability of pulling a prize -/
def prizeProbability (favorable : ℕ) (unfavorable : ℕ) : ℚ :=
  (favorable : ℚ) / (favorable + unfavorable : ℚ)

theorem prize_probability_after_addition (initial_favorable : ℕ) (initial_unfavorable : ℕ) 
  (h_initial_odds : initial_favorable = 5 ∧ initial_unfavorable = 6) 
  (added_prizes : ℕ) (h_added_prizes : added_prizes = 2) :
  prizeProbability (initial_favorable + added_prizes) initial_unfavorable = 7 / 13 := by
  sorry

#check prize_probability_after_addition

end NUMINAMATH_CALUDE_prize_probability_after_addition_l3745_374537


namespace NUMINAMATH_CALUDE_polynomial_sum_l3745_374579

-- Define the polynomials
def f (x : ℝ) : ℝ := -3 * x^3 - 3 * x^2 + x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 5 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) : f x + g x + h x = -3 * x^3 - 4 * x^2 + 11 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3745_374579


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3745_374572

open Set

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_A_complement_B :
  A ∩ (𝒰 \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3745_374572


namespace NUMINAMATH_CALUDE_nine_digit_increasing_integers_mod_1000_l3745_374536

/-- The number of ways to select 9 items from 10 items with replacement and order matters -/
def M : ℕ := Nat.choose 18 9

/-- The theorem to prove -/
theorem nine_digit_increasing_integers_mod_1000 :
  M % 1000 = 620 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_increasing_integers_mod_1000_l3745_374536


namespace NUMINAMATH_CALUDE_tennis_players_count_l3745_374538

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (both : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 18 →
  both = 9 →
  neither = 2 →
  ∃ tennis : ℕ, tennis = 19 ∧ 
    total = badminton + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_count_l3745_374538


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l3745_374598

theorem sphere_volume_surface_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 →
  (4/3 * π * r₁^3) / (4/3 * π * r₂^3) = 8 →
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l3745_374598


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3745_374589

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0 → a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3745_374589


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3745_374519

theorem ceiling_floor_difference (x : ℝ) 
  (h : ⌈x⌉ - ⌊x⌋ = 2) : 
  3 * (⌈x⌉ - x) = 6 - 3 * (x - ⌊x⌋) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3745_374519


namespace NUMINAMATH_CALUDE_circle_chords_with_equal_sums_l3745_374510

/-- Given 2^500 points on a circle labeled 1 to 2^500, there exist 100 pairwise disjoint chords
    such that the sums of the labels at their endpoints are all equal. -/
theorem circle_chords_with_equal_sums :
  ∀ (labeling : Fin (2^500) → Fin (2^500)),
  ∃ (chords : Finset (Fin (2^500) × Fin (2^500))),
    (chords.card = 100) ∧
    (∀ (c1 c2 : Fin (2^500) × Fin (2^500)), c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
      (c1.1 ≠ c2.1 ∧ c1.1 ≠ c2.2 ∧ c1.2 ≠ c2.1 ∧ c1.2 ≠ c2.2)) ∧
    (∃ (sum : Nat), ∀ (c : Fin (2^500) × Fin (2^500)), c ∈ chords → 
      (labeling c.1).val + (labeling c.2).val = sum) :=
by sorry

end NUMINAMATH_CALUDE_circle_chords_with_equal_sums_l3745_374510


namespace NUMINAMATH_CALUDE_infinite_solutions_when_m_is_two_l3745_374551

theorem infinite_solutions_when_m_is_two :
  ∃ (m : ℝ), ∀ (x : ℝ), m^2 * x + m * (1 - x) - 2 * (1 + x) = 0 → 
  (m = 2 ∧ ∀ (y : ℝ), m^2 * y + m * (1 - y) - 2 * (1 + y) = 0) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_when_m_is_two_l3745_374551


namespace NUMINAMATH_CALUDE_unintended_texts_per_week_l3745_374576

theorem unintended_texts_per_week 
  (old_daily_texts : ℕ) 
  (new_daily_texts : ℕ) 
  (days_in_week : ℕ) 
  (h1 : old_daily_texts = 20)
  (h2 : new_daily_texts = 55)
  (h3 : days_in_week = 7) :
  (new_daily_texts - old_daily_texts) * days_in_week = 245 :=
by sorry

end NUMINAMATH_CALUDE_unintended_texts_per_week_l3745_374576


namespace NUMINAMATH_CALUDE_decimal_subtraction_l3745_374578

theorem decimal_subtraction :
  let largest_three_digit := 0.999
  let smallest_four_digit := 0.0001
  largest_three_digit - smallest_four_digit = 0.9989 := by
  sorry

end NUMINAMATH_CALUDE_decimal_subtraction_l3745_374578


namespace NUMINAMATH_CALUDE_ryan_spanish_hours_l3745_374515

/-- Ryan's daily study hours -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  spanish : ℕ

/-- Ryan's study schedule satisfies the given conditions -/
def validSchedule (h : StudyHours) : Prop :=
  h.english = 7 ∧ h.chinese = 2 ∧ h.english = h.spanish + 3

theorem ryan_spanish_hours (h : StudyHours) (hvalid : validSchedule h) : h.spanish = 4 := by
  sorry

end NUMINAMATH_CALUDE_ryan_spanish_hours_l3745_374515


namespace NUMINAMATH_CALUDE_sector_max_area_l3745_374590

/-- Given a sector of a circle with radius R, central angle α, and fixed perimeter c,
    the maximum area of the sector is c²/16. -/
theorem sector_max_area (R α c : ℝ) (h_pos_R : R > 0) (h_pos_α : α > 0) (h_pos_c : c > 0)
  (h_perimeter : c = 2 * R + R * α) :
  ∃ (A : ℝ), A ≤ c^2 / 16 ∧ 
  (∀ (R' α' : ℝ), R' > 0 → α' > 0 → c = 2 * R' + R' * α' → 
    (1/2) * R' * R' * α' ≤ A) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3745_374590


namespace NUMINAMATH_CALUDE_sallys_cards_l3745_374562

/-- Sally's card counting problem -/
theorem sallys_cards (initial : ℕ) (dans_gift : ℕ) (sallys_purchase : ℕ) : 
  initial = 27 → dans_gift = 41 → sallys_purchase = 20 → 
  initial + dans_gift + sallys_purchase = 88 := by
  sorry

end NUMINAMATH_CALUDE_sallys_cards_l3745_374562


namespace NUMINAMATH_CALUDE_root_in_interval_l3745_374548

def f (x : ℝ) := 2*x + 3*x - 7

theorem root_in_interval :
  ∃ r ∈ Set.Ioo 1 2, f r = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3745_374548


namespace NUMINAMATH_CALUDE_cindy_same_color_prob_l3745_374511

/-- Represents the number of marbles of each color in the box -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (mc : MarbleCount) : ℕ := mc.red + mc.green + mc.yellow

/-- Represents the number of marbles drawn by each person -/
structure DrawCounts where
  alice : ℕ
  bob : ℕ
  cindy : ℕ

/-- Calculates the probability of Cindy getting 3 marbles of the same color -/
noncomputable def probCindySameColor (mc : MarbleCount) (dc : DrawCounts) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem cindy_same_color_prob :
  let initial_marbles : MarbleCount := ⟨2, 2, 4⟩
  let draw_counts : DrawCounts := ⟨2, 3, 3⟩
  probCindySameColor initial_marbles draw_counts = 13 / 140 :=
sorry

end NUMINAMATH_CALUDE_cindy_same_color_prob_l3745_374511


namespace NUMINAMATH_CALUDE_mean_calculation_l3745_374529

theorem mean_calculation (x : ℝ) :
  (28 + x + 50 + 78 + 104) / 5 = 62 →
  (48 + 62 + 98 + 124 + x) / 5 = 76.4 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l3745_374529


namespace NUMINAMATH_CALUDE_cos_function_identity_l3745_374525

theorem cos_function_identity (f : ℝ → ℝ) (x : ℝ) 
  (h : ∀ x, f (Real.sin x) = 2 - Real.cos (2 * x)) : 
  f (Real.cos x) = 2 + Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_function_identity_l3745_374525


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3745_374575

/-- Given three lines in the xy-plane:
    L₁: x + y - 2 = 0
    L₂: 3x + 2y - 5 = 0
    L₃: 3x + 4y - 12 = 0
    Prove that the line L: 4x - 3y - 1 = 0 passes through the intersection of L₁ and L₂,
    and is perpendicular to L₃. -/
theorem intersection_and_perpendicular_line 
  (L₁ : Set (ℝ × ℝ) := {p | p.1 + p.2 - 2 = 0})
  (L₂ : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 - 5 = 0})
  (L₃ : Set (ℝ × ℝ) := {p | 3 * p.1 + 4 * p.2 - 12 = 0})
  (L : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 - 1 = 0}) :
  (∃ p, p ∈ L₁ ∩ L₂ ∧ p ∈ L) ∧
  (∀ p q : ℝ × ℝ, p ≠ q → p ∈ L → q ∈ L → p ∈ L₃ → q ∈ L₃ → 
    (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3745_374575


namespace NUMINAMATH_CALUDE_mango_juice_cost_l3745_374558

/-- The cost of a big bottle of mango juice in pesetas -/
def big_bottle_cost : ℕ := 2700

/-- The volume of a big bottle in ounces -/
def big_bottle_volume : ℕ := 30

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℕ := 6

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℕ := 600

/-- The amount saved by buying a big bottle instead of equivalent small bottles in pesetas -/
def saving : ℕ := 300

theorem mango_juice_cost :
  big_bottle_cost = 
    (big_bottle_volume / small_bottle_volume) * small_bottle_cost - saving :=
by sorry

end NUMINAMATH_CALUDE_mango_juice_cost_l3745_374558


namespace NUMINAMATH_CALUDE_simplify_fraction_l3745_374595

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3745_374595


namespace NUMINAMATH_CALUDE_BC_length_l3745_374500

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
axiom right_triangle_ABC : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
axiom right_triangle_ABD : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
axiom AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 45
axiom BD_length : Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 52

-- Theorem statement
theorem BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 1079 := by
  sorry

end NUMINAMATH_CALUDE_BC_length_l3745_374500


namespace NUMINAMATH_CALUDE_innings_count_l3745_374594

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  average : ℝ
  highestScore : ℕ
  scoreDifference : ℕ
  averageExcludingExtremes : ℝ

/-- Calculates the number of innings played by a batsman given their stats -/
def calculateInnings (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that for the given batsman stats, the number of innings is 104 -/
theorem innings_count (stats : BatsmanStats) 
  (h1 : stats.average = 62)
  (h2 : stats.highestScore = 225)
  (h3 : stats.scoreDifference = 150)
  (h4 : stats.averageExcludingExtremes = 58) :
  calculateInnings stats = 104 := by
  sorry

end NUMINAMATH_CALUDE_innings_count_l3745_374594


namespace NUMINAMATH_CALUDE_triangle_side_length_l3745_374534

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →   -- Area of the triangle is √3
  (B = π/3) →                                 -- Angle B is 60°
  (a^2 + c^2 = 3*a*c) →                        -- Given condition
  (b = 2 * Real.sqrt 2) :=                     -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3745_374534


namespace NUMINAMATH_CALUDE_coin_distribution_rotations_l3745_374509

/-- Represents the coin distribution problem on a round table. -/
structure CoinDistribution where
  n : ℕ  -- number of sectors and players
  m : ℕ  -- number of rotations
  h_n_ge_4 : n ≥ 4

  /-- Player 1 received 74 fewer coins than player 4 -/
  h_player1_4 : ∃ (c1 c4 : ℕ), c4 - c1 = 74

  /-- Player 2 received 50 fewer coins than player 3 -/
  h_player2_3 : ∃ (c2 c3 : ℕ), c3 - c2 = 50

  /-- Player 4 received 3 coins twice as often as 2 coins -/
  h_player4_3_2 : ∃ (t2 t3 : ℕ), t3 = 2 * t2

  /-- Player 4 received 3 coins half as often as 1 coin -/
  h_player4_3_1 : ∃ (t1 t3 : ℕ), t3 = t1 / 2

/-- The number of rotations in the coin distribution problem is 69. -/
theorem coin_distribution_rotations (cd : CoinDistribution) : cd.m = 69 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_rotations_l3745_374509


namespace NUMINAMATH_CALUDE_independence_implies_a_minus_b_eq_neg_two_l3745_374541

theorem independence_implies_a_minus_b_eq_neg_two :
  ∀ (a b : ℝ), 
  (∀ x : ℝ, ∃ c : ℝ, ∀ y : ℝ, x^2 + a*x - (b*y^2 - y - 3) = c) →
  a - b = -2 :=
by sorry

end NUMINAMATH_CALUDE_independence_implies_a_minus_b_eq_neg_two_l3745_374541


namespace NUMINAMATH_CALUDE_no_common_integers_satisfying_condition_l3745_374512

theorem no_common_integers_satisfying_condition : 
  ¬∃ i : ℤ, 10 ≤ i ∧ i ≤ 30 ∧ i^2 - 5*i - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_integers_satisfying_condition_l3745_374512


namespace NUMINAMATH_CALUDE_percentage_silver_cars_l3745_374571

/-- Calculates the percentage of silver cars after a new shipment -/
theorem percentage_silver_cars (initial_cars : ℕ) (initial_silver_percentage : ℚ) 
  (new_cars : ℕ) (new_non_silver_percentage : ℚ) :
  initial_cars = 40 →
  initial_silver_percentage = 1/5 →
  new_cars = 80 →
  new_non_silver_percentage = 1/2 →
  let initial_silver := initial_cars * initial_silver_percentage
  let new_silver := new_cars * (1 - new_non_silver_percentage)
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_cars
  (total_silver / total_cars : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_silver_cars_l3745_374571


namespace NUMINAMATH_CALUDE_event_attendees_l3745_374543

/-- Represents the number of men at the event -/
def num_men : ℕ := 15

/-- Represents the number of women each man danced with -/
def dances_per_man : ℕ := 4

/-- Represents the number of men each woman danced with -/
def dances_per_woman : ℕ := 3

/-- Calculates the number of women at the event -/
def num_women : ℕ := (num_men * dances_per_man) / dances_per_woman

theorem event_attendees :
  num_women = 20 := by
  sorry

end NUMINAMATH_CALUDE_event_attendees_l3745_374543


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3745_374533

theorem stratified_sample_size 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_female : ℕ) 
  (h1 : total_male = 42) 
  (h2 : total_female = 30) 
  (h3 : sample_female = 5) :
  ∃ (sample_male : ℕ), 
    (sample_male : ℚ) / (sample_female : ℚ) = (total_male : ℚ) / (total_female : ℚ) ∧
    sample_male + sample_female = 12 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3745_374533


namespace NUMINAMATH_CALUDE_output_after_year_formula_l3745_374549

/-- Calculates the output after 12 months given an initial output and monthly growth rate -/
def outputAfterYear (a : ℝ) (p : ℝ) : ℝ := a * (1 + p) ^ 12

/-- Theorem stating that the output after 12 months is equal to a(1+p)^12 -/
theorem output_after_year_formula (a : ℝ) (p : ℝ) :
  outputAfterYear a p = a * (1 + p) ^ 12 := by sorry

end NUMINAMATH_CALUDE_output_after_year_formula_l3745_374549


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l3745_374582

theorem chicken_wings_distribution (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) :
  num_friends = 4 →
  initial_wings = 9 →
  additional_wings = 7 →
  (initial_wings + additional_wings) % num_friends = 0 →
  (initial_wings + additional_wings) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l3745_374582


namespace NUMINAMATH_CALUDE_actual_daily_length_is_72_required_daily_increase_at_least_36_l3745_374527

/-- Represents the renovation of a pipe network --/
structure PipeRenovation where
  totalLength : ℝ
  originalDailyLength : ℝ
  efficiencyIncrease : ℝ
  daysAheadOfSchedule : ℝ
  constructedDays : ℝ
  maxTotalDays : ℝ

/-- Calculates the actual daily renovation length --/
def actualDailyLength (pr : PipeRenovation) : ℝ :=
  pr.originalDailyLength * (1 + pr.efficiencyIncrease)

/-- Theorem for the actual daily renovation length --/
theorem actual_daily_length_is_72 (pr : PipeRenovation)
  (h1 : pr.totalLength = 3600)
  (h2 : pr.efficiencyIncrease = 0.2)
  (h3 : pr.daysAheadOfSchedule = 10)
  (h4 : pr.totalLength / pr.originalDailyLength - pr.totalLength / (actualDailyLength pr) = pr.daysAheadOfSchedule) :
  actualDailyLength pr = 72 := by sorry

/-- Theorem for the required increase in daily renovation length --/
theorem required_daily_increase_at_least_36 (pr : PipeRenovation)
  (h1 : pr.totalLength = 3600)
  (h2 : actualDailyLength pr = 72)
  (h3 : pr.constructedDays = 20)
  (h4 : pr.maxTotalDays = 40) :
  ∃ m : ℝ, m ≥ 36 ∧ (pr.maxTotalDays - pr.constructedDays) * (actualDailyLength pr + m) ≥ pr.totalLength - actualDailyLength pr * pr.constructedDays := by sorry

end NUMINAMATH_CALUDE_actual_daily_length_is_72_required_daily_increase_at_least_36_l3745_374527


namespace NUMINAMATH_CALUDE_mary_zoom_time_l3745_374546

def total_time (mac_download : ℕ) (windows_download_factor : ℕ) 
               (audio_glitch_duration : ℕ) (audio_glitch_count : ℕ)
               (video_glitch_duration : ℕ) : ℕ :=
  let windows_download := mac_download * windows_download_factor
  let total_download := mac_download + windows_download
  let audio_glitch_time := audio_glitch_duration * audio_glitch_count
  let total_glitch_time := audio_glitch_time + video_glitch_duration
  let glitch_free_time := 2 * total_glitch_time
  total_download + total_glitch_time + glitch_free_time

theorem mary_zoom_time : 
  total_time 10 3 4 2 6 = 82 := by
  sorry

end NUMINAMATH_CALUDE_mary_zoom_time_l3745_374546


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l3745_374517

/-- The sum of the series defined by the nth term 1/((n+1)(n+2)) - 1/((n+2)(n+3)) for n ≥ 1 is equal to 1/2. -/
theorem series_sum_equals_half :
  (∑' n : ℕ, (1 : ℝ) / ((n + 1) * (n + 2)) - 1 / ((n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l3745_374517


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3745_374554

/-- The area of a triangle with sides 16, 30, and 34 is 240 -/
theorem triangle_area : ℝ → Prop :=
  fun a : ℝ =>
    let s1 : ℝ := 16
    let s2 : ℝ := 30
    let s3 : ℝ := 34
    (s1 * s1 + s2 * s2 = s3 * s3) →  -- Pythagorean theorem condition
    (a = (1 / 2) * s1 * s2) →        -- Area formula for right triangle
    a = 240

/-- Proof of the theorem -/
theorem triangle_area_proof : triangle_area 240 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3745_374554


namespace NUMINAMATH_CALUDE_m_range_l3745_374528

def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x - 1| > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(7 - 3*m))^x > (-(7 - 3*m))^y

theorem m_range : 
  (∃ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) ∧ 
  (∀ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m) → m ∈ Set.Icc 1 2) ∧
  (∀ m : ℝ, m ∈ Set.Icc 1 2 → (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3745_374528


namespace NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_l3745_374521

-- Sequence 1
def sequence1 (n : ℕ) : ℚ :=
  (-5^n + (-1)^(n-1) * 3 * 2^(n+1)) / (2 * 5^n + (-1)^(n-1) * 2^(n+1))

def sequence1_recurrence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ ∀ n, n ≥ 1 → a (n+1) = (a n + 3) / (2 * a n - 4)

theorem sequence1_correct :
  sequence1_recurrence sequence1 := by sorry

-- Sequence 2
def sequence2 (n : ℕ) : ℚ :=
  (6*n - 11) / (3*n - 4)

def sequence2_recurrence (a : ℕ → ℚ) : Prop :=
  a 1 = 5 ∧ ∀ n, n ≥ 1 → a (n+1) = (a n - 4) / (a n - 3)

theorem sequence2_correct :
  sequence2_recurrence sequence2 := by sorry

end NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_l3745_374521


namespace NUMINAMATH_CALUDE_middle_integer_of_consecutive_sum_l3745_374547

theorem middle_integer_of_consecutive_sum (n : ℤ) : 
  (n - 1) + n + (n + 1) = 180 → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_middle_integer_of_consecutive_sum_l3745_374547


namespace NUMINAMATH_CALUDE_photo_arrangements_l3745_374569

def number_of_students : ℕ := 7
def number_of_bound_students : ℕ := 2
def number_of_separated_students : ℕ := 2

def arrangements (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem photo_arrangements :
  let bound_ways := number_of_bound_students
  let remaining_elements := number_of_students - number_of_bound_students - number_of_separated_students + 1
  let gaps := remaining_elements + 1
  bound_ways * arrangements remaining_elements remaining_elements * arrangements gaps number_of_separated_students = 960 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l3745_374569


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3745_374565

theorem sqrt_equation_solution (a b : ℝ) : 
  Real.sqrt (a - 5) + Real.sqrt (5 - a) = b + 3 → 
  a = 5 ∧ (Real.sqrt (a^2 - b^2) = 4 ∨ Real.sqrt (a^2 - b^2) = -4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3745_374565


namespace NUMINAMATH_CALUDE_function_properties_l3745_374505

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f x * f y = (f (x + y) + 2 * f (x - y)) / 3)
  (h2 : ∀ x : ℝ, f x ≠ 0) :
  (f 0 = 1) ∧ (∀ x : ℝ, f x = f (-x)) := by sorry

end NUMINAMATH_CALUDE_function_properties_l3745_374505


namespace NUMINAMATH_CALUDE_jafari_candy_count_l3745_374591

theorem jafari_candy_count (total candy_taquon candy_mack : ℕ) 
  (h1 : total = candy_taquon + candy_mack + (total - candy_taquon - candy_mack))
  (h2 : candy_taquon = 171)
  (h3 : candy_mack = 171)
  (h4 : total = 418) :
  total - candy_taquon - candy_mack = 76 := by
sorry

end NUMINAMATH_CALUDE_jafari_candy_count_l3745_374591


namespace NUMINAMATH_CALUDE_truck_distance_l3745_374566

theorem truck_distance (north_distance east_distance : ℝ) 
  (h1 : north_distance = 40)
  (h2 : east_distance = 30) :
  Real.sqrt (north_distance ^ 2 + east_distance ^ 2) = 50 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l3745_374566


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3745_374507

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, |a + b| > 1 → |a| + |b| > 1) ∧
  (∃ a b : ℝ, |a| + |b| > 1 ∧ |a + b| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3745_374507


namespace NUMINAMATH_CALUDE_ellipse_through_six_points_l3745_374545

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a point lies on an ellipse with center (h, k), semi-major axis a, and semi-minor axis b -/
def onEllipse (p : Point) (h k a b : ℝ) : Prop :=
  ((p.x - h) ^ 2 / a ^ 2) + ((p.y - k) ^ 2 / b ^ 2) = 1

theorem ellipse_through_six_points :
  let p1 : Point := ⟨-3, 2⟩
  let p2 : Point := ⟨0, 0⟩
  let p3 : Point := ⟨0, 4⟩
  let p4 : Point := ⟨6, 0⟩
  let p5 : Point := ⟨6, 4⟩
  let p6 : Point := ⟨-3, 0⟩
  let points := [p1, p2, p3, p4, p5, p6]
  (∀ (a b c : Point), a ∈ points → b ∈ points → c ∈ points → a ≠ b → b ≠ c → a ≠ c → ¬collinear a b c) →
  ∃ (h k a b : ℝ), 
    a = 6 ∧ 
    b = 1 ∧ 
    (∀ p ∈ points, onEllipse p h k a b) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_through_six_points_l3745_374545


namespace NUMINAMATH_CALUDE_possible_m_values_l3745_374501

def A : Set ℝ := {x | x^2 - 9*x - 10 = 0}

def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values :
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ∈ ({0, 1, -(1/10)} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l3745_374501


namespace NUMINAMATH_CALUDE_remaining_area_formula_l3745_374581

/-- The remaining area of a rectangle with a hole -/
def remaining_area (x : ℝ) : ℝ :=
  (2*x + 5) * (x + 8) - (3*x - 2) * (x + 1)

/-- Theorem: The remaining area is equal to -x^2 + 20x + 42 -/
theorem remaining_area_formula (x : ℝ) :
  remaining_area x = -x^2 + 20*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_formula_l3745_374581


namespace NUMINAMATH_CALUDE_b_initial_investment_l3745_374583

/-- Given A's investment and doubling conditions, proves B's initial investment --/
theorem b_initial_investment 
  (a_initial : ℕ) 
  (a_doubles_after_six_months : Bool) 
  (equal_yearly_investment : Bool) : ℕ :=
by
  -- Assuming a_initial = 3000, a_doubles_after_six_months = true, and equal_yearly_investment = true
  sorry

#check b_initial_investment

end NUMINAMATH_CALUDE_b_initial_investment_l3745_374583


namespace NUMINAMATH_CALUDE_distinct_reciprocals_inequality_l3745_374523

theorem distinct_reciprocals_inequality (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_sum : 2 * b = a + c) : 
  2 / b ≠ 1 / a + 1 / c := by
sorry

end NUMINAMATH_CALUDE_distinct_reciprocals_inequality_l3745_374523


namespace NUMINAMATH_CALUDE_sum_of_products_l3745_374508

theorem sum_of_products (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 5*x₂ + 10*x₃ + 17*x₄ + 26*x₅ + 37*x₆ + 50*x₇ + 65*x₈ = 2)
  (eq2 : 5*x₁ + 10*x₂ + 17*x₃ + 26*x₄ + 37*x₅ + 50*x₆ + 65*x₇ + 82*x₈ = 14)
  (eq3 : 10*x₁ + 17*x₂ + 26*x₃ + 37*x₄ + 50*x₅ + 65*x₆ + 82*x₇ + 101*x₈ = 140) :
  17*x₁ + 26*x₂ + 37*x₃ + 50*x₄ + 65*x₅ + 82*x₆ + 101*x₇ + 122*x₈ = 608 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l3745_374508


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l3745_374544

/-- The number of balls -/
def n : ℕ := 30

/-- The number of bins -/
def m : ℕ := 6

/-- The probability of one bin having 6 balls, one having 3 balls, and four having 5 balls each -/
noncomputable def p' : ℝ := sorry

/-- The probability of all bins having exactly 5 balls -/
noncomputable def q' : ℝ := sorry

/-- The theorem stating that the ratio of p' to q' is 5 -/
theorem ball_distribution_ratio : p' / q' = 5 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l3745_374544


namespace NUMINAMATH_CALUDE_divisible_by_13_with_sqrt_between_24_and_24_5_verify_585_and_598_l3745_374553

theorem divisible_by_13_with_sqrt_between_24_and_24_5 : 
  ∃ (n : ℕ), n > 0 ∧ n % 13 = 0 ∧ 24 < Real.sqrt n ∧ Real.sqrt n < 24.5 :=
by
  sorry

theorem verify_585_and_598 : 
  (585 > 0 ∧ 585 % 13 = 0 ∧ 24 < Real.sqrt 585 ∧ Real.sqrt 585 < 24.5) ∧
  (598 > 0 ∧ 598 % 13 = 0 ∧ 24 < Real.sqrt 598 ∧ Real.sqrt 598 < 24.5) :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_13_with_sqrt_between_24_and_24_5_verify_585_and_598_l3745_374553


namespace NUMINAMATH_CALUDE_people_to_lift_car_l3745_374531

theorem people_to_lift_car : ℕ :=
  let people_for_car : ℕ := sorry
  let people_for_truck : ℕ := 2 * people_for_car
  have h1 : 6 * people_for_car + 3 * people_for_truck = 60 := by sorry
  have h2 : people_for_car = 5 := by sorry
  5

#check people_to_lift_car

end NUMINAMATH_CALUDE_people_to_lift_car_l3745_374531


namespace NUMINAMATH_CALUDE_five_million_squared_l3745_374592

theorem five_million_squared (five_million : ℕ) (h : five_million = 5 * 10^6) :
  five_million^2 = 25 * 10^12 := by
  sorry

end NUMINAMATH_CALUDE_five_million_squared_l3745_374592


namespace NUMINAMATH_CALUDE_sum_due_proof_l3745_374586

/-- Represents the relationship between banker's discount, true discount, and face value -/
def discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + (td * bd / fv)

/-- Proves that given a banker's discount of 36 and a true discount of 30,
    the face value (sum due) is 180 -/
theorem sum_due_proof :
  ∃ (fv : ℚ), discount_relation 36 30 fv ∧ fv = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_proof_l3745_374586


namespace NUMINAMATH_CALUDE_age_difference_l3745_374577

/-- Proves that the age difference between a man and his student is 26 years -/
theorem age_difference (student_age man_age : ℕ) : 
  student_age = 24 →
  man_age + 2 = 2 * (student_age + 2) →
  man_age - student_age = 26 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3745_374577


namespace NUMINAMATH_CALUDE_quadratic_rational_roots_unique_b_l3745_374559

theorem quadratic_rational_roots_unique_b : 
  ∃! b : ℕ+, (∃ x y : ℚ, 3 * x^2 + 6 * x + b.val = 0 ∧ 3 * y^2 + 6 * y + b.val = 0 ∧ x ≠ y) ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_roots_unique_b_l3745_374559


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l3745_374502

/-- Represents Sheila's work schedule and earnings --/
structure SheilaWork where
  hourly_rate : ℕ
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  days_mon_wed_fri : ℕ
  days_tue_thu : ℕ

/-- Calculates Sheila's weekly earnings --/
def weekly_earnings (s : SheilaWork) : ℕ :=
  s.hourly_rate * (s.hours_mon_wed_fri * s.days_mon_wed_fri + s.hours_tue_thu * s.days_tue_thu)

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  ∃ (s : SheilaWork),
    s.hourly_rate = 13 ∧
    s.hours_mon_wed_fri = 8 ∧
    s.hours_tue_thu = 6 ∧
    s.days_mon_wed_fri = 3 ∧
    s.days_tue_thu = 2 ∧
    weekly_earnings s = 468 := by
  sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l3745_374502


namespace NUMINAMATH_CALUDE_tangent_sum_l3745_374522

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l3745_374522


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3745_374599

-- Define the universal set U
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 7}

-- Define set A
def A : Set ℤ := {1, 3, 5, 7}

-- Define set B
def B : Set ℤ := {2, 4, 5}

-- Theorem statement
theorem intersection_complement_equals_set : B ∩ (U \ A) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3745_374599


namespace NUMINAMATH_CALUDE_coefficient_x5y_in_expansion_l3745_374597

/-- The coefficient of x^5y in the expansion of (x-2y)^5(x+y) -/
def coefficient_x5y : ℤ := -9

/-- The expansion of (x-2y)^5(x+y) -/
def expansion (x y : ℚ) : ℚ := (x - 2*y)^5 * (x + y)

theorem coefficient_x5y_in_expansion :
  coefficient_x5y = (
    -- Extract the coefficient of x^5y from the expansion
    -- This part is left unimplemented as it requires complex polynomial manipulation
    sorry
  ) := by sorry

end NUMINAMATH_CALUDE_coefficient_x5y_in_expansion_l3745_374597


namespace NUMINAMATH_CALUDE_problem_solution_l3745_374550

def f (a : ℝ) (x : ℝ) : ℝ := |x| * (x - a)

theorem problem_solution :
  (∀ x, f a x = -f a (-x)) → a = 0 ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f a x ≤ f a y) → a ≤ 0 ∧
  ∃ a, a < 0 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1/2 → f a x ≤ 2) ∧
     (∃ x, -1 ≤ x ∧ x ≤ 1/2 ∧ f a x = 2) ∧
     a = -3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3745_374550


namespace NUMINAMATH_CALUDE_range_of_a_l3745_374596

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → a < x + 1/x) → a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3745_374596


namespace NUMINAMATH_CALUDE_people_per_car_l3745_374520

/-- Proves that if 63 people are equally divided among 9 cars, then the number of people in each car is 7. -/
theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (people_per_car : ℕ) 
  (h1 : total_people = 63) 
  (h2 : num_cars = 9) 
  (h3 : total_people = num_cars * people_per_car) : 
  people_per_car = 7 := by
  sorry

end NUMINAMATH_CALUDE_people_per_car_l3745_374520


namespace NUMINAMATH_CALUDE_seven_strip_trapezoid_shaded_area_l3745_374539

/-- Represents a trapezoid divided into equal width strips -/
structure StripTrapezoid where
  numStrips : ℕ
  numShaded : ℕ
  h_pos : 0 < numStrips
  h_shaded : numShaded ≤ numStrips

/-- The fraction of shaded area in a strip trapezoid -/
def shadedAreaFraction (t : StripTrapezoid) : ℚ :=
  t.numShaded / t.numStrips

/-- Theorem: In a trapezoid divided into 7 strips with 4 shaded, the shaded area is 4/7 of the total area -/
theorem seven_strip_trapezoid_shaded_area :
  let t : StripTrapezoid := ⟨7, 4, by norm_num, by norm_num⟩
  shadedAreaFraction t = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_strip_trapezoid_shaded_area_l3745_374539


namespace NUMINAMATH_CALUDE_floor_e_equals_two_l3745_374503

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_e_equals_two_l3745_374503


namespace NUMINAMATH_CALUDE_some_value_is_zero_l3745_374580

theorem some_value_is_zero (x y w : ℝ) (some_value : ℝ) 
  (h1 : some_value + 3 / x = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 1 / 2) :
  some_value = 0 := by
sorry

end NUMINAMATH_CALUDE_some_value_is_zero_l3745_374580


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3745_374532

/-- The axis of symmetry for the parabola x = -4y² is x = 1/16 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun y ↦ -4 * y^2
  ∃ x₀ : ℝ, x₀ = 1/16 ∧ ∀ y : ℝ, f y = f (-y) → x₀ = f y :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3745_374532


namespace NUMINAMATH_CALUDE_area_enclosed_l3745_374570

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => fun x => |x|
  | k + 1 => fun x => |f k x - (k + 1)|

theorem area_enclosed (n : ℕ) : 
  ∃ (a : ℝ), a > 0 ∧ 
  (∫ (x : ℝ) in -a..a, f n x) = (4 * n^3 + 6 * n^2 - 1 + (-1)^n) / 8 :=
sorry

end NUMINAMATH_CALUDE_area_enclosed_l3745_374570


namespace NUMINAMATH_CALUDE_celebrity_baby_picture_matching_probability_l3745_374524

theorem celebrity_baby_picture_matching_probability :
  ∀ (n : ℕ), n = 5 →
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 120 :=
by sorry

end NUMINAMATH_CALUDE_celebrity_baby_picture_matching_probability_l3745_374524


namespace NUMINAMATH_CALUDE_train_acceleration_equation_l3745_374568

theorem train_acceleration_equation 
  (v : ℝ) (s : ℝ) (x : ℝ) 
  (h1 : v > 0) 
  (h2 : s > 0) 
  (h3 : x > v) :
  s / (x - v) = (s + 50) / x :=
by sorry

end NUMINAMATH_CALUDE_train_acceleration_equation_l3745_374568


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l3745_374561

/-- Represents the carpet configuration with shaded squares -/
structure CarpetConfig where
  carpet_side : ℝ
  large_square_side : ℝ
  small_square_side : ℝ
  large_square_count : ℕ
  small_square_count : ℕ

/-- Calculates the total shaded area of the carpet -/
def total_shaded_area (config : CarpetConfig) : ℝ :=
  config.large_square_count * config.large_square_side^2 +
  config.small_square_count * config.small_square_side^2

/-- Theorem stating the total shaded area of the carpet with given conditions -/
theorem carpet_shaded_area :
  ∀ (config : CarpetConfig),
    config.carpet_side = 12 →
    config.carpet_side / config.large_square_side = 4 →
    config.large_square_side / config.small_square_side = 3 →
    config.large_square_count = 1 →
    config.small_square_count = 8 →
    total_shaded_area config = 17 := by
  sorry


end NUMINAMATH_CALUDE_carpet_shaded_area_l3745_374561


namespace NUMINAMATH_CALUDE_min_difference_of_bounds_l3745_374552

-- Define the arithmetic-geometric sequence
def a (n : ℕ) : ℚ := (4/3) * (-1/3)^(n-1)

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := 1 - (-1/3)^n

-- Define the function f(n) = S(n) - 1/S(n)
def f (n : ℕ) : ℚ := S n - 1 / (S n)

-- Theorem statement
theorem min_difference_of_bounds (A B : ℚ) :
  (∀ n : ℕ, n ≥ 1 → A ≤ f n ∧ f n ≤ B) →
  B - A ≥ 59/72 :=
sorry

end NUMINAMATH_CALUDE_min_difference_of_bounds_l3745_374552


namespace NUMINAMATH_CALUDE_fraction_simplification_l3745_374587

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3745_374587


namespace NUMINAMATH_CALUDE_cube_root_of_27_l3745_374513

theorem cube_root_of_27 : 
  {z : ℂ | z^3 = 27} = {3, (-3 + 3*Complex.I*Real.sqrt 3)/2, (-3 - 3*Complex.I*Real.sqrt 3)/2} := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_27_l3745_374513


namespace NUMINAMATH_CALUDE_angle_CAD_measure_l3745_374574

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a pentagon -/
structure Pentagon :=
  (B : Point)
  (C : Point)
  (D : Point)
  (E : Point)
  (G : Point)

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Checks if a pentagon is regular -/
def is_regular_pentagon (p : Pentagon) : Prop := sorry

/-- Calculates the angle between three points in degrees -/
def angle_deg (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem angle_CAD_measure 
  (t : Triangle) 
  (p : Pentagon) 
  (h1 : is_equilateral t)
  (h2 : is_regular_pentagon p)
  (h3 : t.B = p.B)
  (h4 : t.C = p.C) :
  angle_deg t.A p.D t.C = 24 := by sorry

end NUMINAMATH_CALUDE_angle_CAD_measure_l3745_374574


namespace NUMINAMATH_CALUDE_apples_per_pie_l3745_374564

theorem apples_per_pie 
  (initial_apples : ℕ) 
  (handed_out : ℕ) 
  (num_pies : ℕ) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out = 19) 
  (h3 : num_pies = 7) :
  (initial_apples - handed_out) / num_pies = 8 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3745_374564


namespace NUMINAMATH_CALUDE_toms_marbles_pairs_l3745_374584

/-- Represents the set of marbles Tom has --/
structure MarbleSet where
  unique_colors : ℕ
  yellow_count : ℕ
  orange_count : ℕ

/-- Calculates the number of distinct pairs of marbles that can be chosen --/
def distinct_pairs (ms : MarbleSet) : ℕ :=
  let yellow_pairs := if ms.yellow_count ≥ 2 then 1 else 0
  let orange_pairs := if ms.orange_count ≥ 2 then 1 else 0
  let diff_color_pairs := ms.unique_colors.choose 2
  let yellow_other_pairs := ms.unique_colors * ms.yellow_count
  let orange_other_pairs := ms.unique_colors * ms.orange_count
  yellow_pairs + orange_pairs + diff_color_pairs + yellow_other_pairs + orange_other_pairs

/-- Theorem stating that Tom's marble set results in 36 distinct pairs --/
theorem toms_marbles_pairs :
  distinct_pairs { unique_colors := 4, yellow_count := 4, orange_count := 3 } = 36 := by
  sorry

end NUMINAMATH_CALUDE_toms_marbles_pairs_l3745_374584


namespace NUMINAMATH_CALUDE_two_digit_numbers_equal_three_times_product_of_digits_l3745_374563

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n = 3 * (n / 10) * (n % 10)} = {15, 24} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_equal_three_times_product_of_digits_l3745_374563


namespace NUMINAMATH_CALUDE_star_equation_two_roots_l3745_374506

/-- Custom binary operation on real numbers -/
def star (a b : ℝ) : ℝ := a * b^2 - b

/-- Theorem stating the condition for the equation 1※x = k to have two distinct real roots -/
theorem star_equation_two_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ star 1 x₁ = k ∧ star 1 x₂ = k) ↔ k > -1/4 :=
sorry

end NUMINAMATH_CALUDE_star_equation_two_roots_l3745_374506


namespace NUMINAMATH_CALUDE_efficient_coin_labeling_theorem_l3745_374567

/-- A coin labeling is a list of 8 positive integers representing coin values in cents -/
def CoinLabeling := List Nat

/-- Checks if a given coin labeling is n-efficient -/
def is_n_efficient (labeling : CoinLabeling) (n : Nat) : Prop :=
  (labeling.length = 8) ∧
  (∀ (k : Nat), 1 ≤ k ∧ k ≤ n → ∃ (buyer_coins seller_coins : List Nat),
    buyer_coins ⊆ labeling.take 4 ∧
    seller_coins ⊆ labeling.drop 4 ∧
    buyer_coins.sum - seller_coins.sum = k)

/-- The maximum n for which an n-efficient labeling exists -/
def max_efficient_n : Nat := 240

/-- Theorem stating the existence of a 240-efficient labeling and that it's the maximum -/
theorem efficient_coin_labeling_theorem :
  (∃ (labeling : CoinLabeling), is_n_efficient labeling max_efficient_n) ∧
  (∀ (n : Nat), n > max_efficient_n → ¬∃ (labeling : CoinLabeling), is_n_efficient labeling n) :=
sorry

end NUMINAMATH_CALUDE_efficient_coin_labeling_theorem_l3745_374567


namespace NUMINAMATH_CALUDE_uncovered_side_length_l3745_374540

/-- Proves that for a rectangular field with given area and fencing length, 
    the length of the uncovered side is as specified. -/
theorem uncovered_side_length 
  (area : ℝ) 
  (fencing_length : ℝ) 
  (h_area : area = 680) 
  (h_fencing : fencing_length = 178) : 
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * width + length = fencing_length ∧ 
    length = 170 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l3745_374540


namespace NUMINAMATH_CALUDE_equation_solution_l3745_374557

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3745_374557
