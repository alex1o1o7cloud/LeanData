import Mathlib

namespace NUMINAMATH_CALUDE_lee_overall_percentage_l3597_359787

theorem lee_overall_percentage (t : ℝ) (h1 : t > 0) : 
  let james_solo := 0.70 * (t / 2)
  let james_total := 0.85 * t
  let together := james_total - james_solo
  let lee_solo := 0.75 * (t / 2)
  lee_solo + together = 0.875 * t := by
sorry

end NUMINAMATH_CALUDE_lee_overall_percentage_l3597_359787


namespace NUMINAMATH_CALUDE_binomial_sum_problem_l3597_359740

theorem binomial_sum_problem (n : ℕ) (M N : ℝ) : 
  M = (5 - 1/2)^n →  -- Sum of coefficients of (5x - 1/√x)^n
  N = 2^n →          -- Sum of binomial coefficients
  M - N = 240 → 
  N = 16 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_problem_l3597_359740


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3597_359764

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ i : ℕ, i ∈ Finset.range 10 → i.succ ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ i : ℕ, i ∈ Finset.range 10 → i.succ ∣ m) → n ≤ m) :=
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3597_359764


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l3597_359797

theorem initial_markup_percentage (C : ℝ) (M : ℝ) : 
  C > 0 →
  let S₁ := C * (1 + M)
  let S₂ := S₁ * 1.25
  let S₃ := S₂ * 0.94
  S₃ = C * 1.41 →
  M = 0.2 := by sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l3597_359797


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l3597_359728

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ 1 → (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l3597_359728


namespace NUMINAMATH_CALUDE_no_solution_condition_l3597_359734

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 * x + 2*a) / (a*x - 2 + a^2) ≥ 0 ∧ a*x + a > 5/4)) ↔ 
  (a ≤ -1/2 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3597_359734


namespace NUMINAMATH_CALUDE_sum_of_radii_l3597_359708

noncomputable section

-- Define the circle radius
def R : ℝ := 5

-- Define the ratios of the sectors
def ratio1 : ℝ := 1
def ratio2 : ℝ := 2
def ratio3 : ℝ := 3

-- Define the base radii of the cones
def r₁ : ℝ := (ratio1 / (ratio1 + ratio2 + ratio3)) * R
def r₂ : ℝ := (ratio2 / (ratio1 + ratio2 + ratio3)) * R
def r₃ : ℝ := (ratio3 / (ratio1 + ratio2 + ratio3)) * R

theorem sum_of_radii : r₁ + r₂ + r₃ = R := by
  sorry

end NUMINAMATH_CALUDE_sum_of_radii_l3597_359708


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3597_359758

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 - 7 * x^2 + d * x - 8
  (g 2 = -8) ∧ (g (-3) = -80) → c = 107/7 ∧ d = -302/7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3597_359758


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l3597_359743

/-- Given a quadratic function f(x) = ax² - 4x + c with range [0, +∞),
    prove that the minimum value of 1/c + 9/a is 3 -/
theorem min_value_quadratic_function (a c : ℝ) (h_pos_a : a > 0) (h_pos_c : c > 0)
  (h_range : Set.range (fun x => a * x^2 - 4*x + c) = Set.Ici 0)
  (h_ac : a * c = 4) :
  (∀ y, 1/c + 9/a ≥ y) ∧ (∃ y, 1/c + 9/a = y) ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l3597_359743


namespace NUMINAMATH_CALUDE_layer_cake_frosting_usage_l3597_359799

/-- Represents the amount of frosting in cans used for different types of baked goods. -/
structure FrostingUsage where
  single_cake : ℚ
  pan_brownies : ℚ
  dozen_cupcakes : ℚ
  layer_cake : ℚ

/-- Represents the quantities of different baked goods to be frosted. -/
structure BakedGoods where
  layer_cakes : ℕ
  dozen_cupcakes : ℕ
  single_cakes : ℕ
  pans_brownies : ℕ

/-- Calculates the total number of cans of frosting needed for a given set of baked goods and frosting usage. -/
def total_frosting (usage : FrostingUsage) (goods : BakedGoods) : ℚ :=
  usage.layer_cake * goods.layer_cakes +
  usage.dozen_cupcakes * goods.dozen_cupcakes +
  usage.single_cake * goods.single_cakes +
  usage.pan_brownies * goods.pans_brownies

/-- The main theorem stating that the amount of frosting used for a layer cake is 1 can. -/
theorem layer_cake_frosting_usage
  (usage : FrostingUsage)
  (goods : BakedGoods)
  (h1 : usage.single_cake = 1/2)
  (h2 : usage.pan_brownies = 1/2)
  (h3 : usage.dozen_cupcakes = 1/2)
  (h4 : goods.layer_cakes = 3)
  (h5 : goods.dozen_cupcakes = 6)
  (h6 : goods.single_cakes = 12)
  (h7 : goods.pans_brownies = 18)
  (h8 : total_frosting usage goods = 21)
  : usage.layer_cake = 1 := by
  sorry

end NUMINAMATH_CALUDE_layer_cake_frosting_usage_l3597_359799


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3597_359701

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (1 + i) = (1 : ℂ) / 2 + (i : ℂ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3597_359701


namespace NUMINAMATH_CALUDE_apples_to_zenny_l3597_359732

theorem apples_to_zenny (total : ℕ) (kept : ℕ) (to_zenny : ℕ) : 
  total = 60 →
  kept = 36 →
  total = to_zenny + (to_zenny + 6) + kept →
  to_zenny = 9 := by sorry

end NUMINAMATH_CALUDE_apples_to_zenny_l3597_359732


namespace NUMINAMATH_CALUDE_circle_op_not_commutative_l3597_359716

/-- Defines the "☉" operation for plane vectors -/
def circle_op (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

/-- Theorem stating that the "☉" operation is not commutative -/
theorem circle_op_not_commutative :
  ∃ (a b : ℝ × ℝ), circle_op a b ≠ circle_op b a :=
sorry

end NUMINAMATH_CALUDE_circle_op_not_commutative_l3597_359716


namespace NUMINAMATH_CALUDE_elliptic_curve_solutions_l3597_359785

theorem elliptic_curve_solutions (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) :
  ∀ (x y : ℤ), y^2 = x^3 - p^2*x ↔ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p ∧ y = 0) ∨ 
    (x = -p ∧ y = 0) ∨ 
    (x = (p^2 + 1)/2 ∧ (y = ((p^2 - 1)/2)*p ∨ y = -((p^2 - 1)/2)*p)) :=
by sorry

end NUMINAMATH_CALUDE_elliptic_curve_solutions_l3597_359785


namespace NUMINAMATH_CALUDE_correct_calculation_l3597_359715

theorem correct_calculation : 3 * Real.sqrt 2 - (Real.sqrt 2) / 2 = (5 / 2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3597_359715


namespace NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l3597_359761

/-- The number of kids who go to camp in Lawrence county during summer break. -/
def kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) : ℕ :=
  total_kids - kids_at_home

/-- Theorem stating the number of kids who go to camp in Lawrence county. -/
theorem lawrence_county_camp_attendance :
  kids_at_camp 1363293 907611 = 455682 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l3597_359761


namespace NUMINAMATH_CALUDE_complex_polynomial_root_l3597_359751

theorem complex_polynomial_root (a b c d : ℤ) : 
  (a * (Complex.I + 3) ^ 5 + b * (Complex.I + 3) ^ 4 + c * (Complex.I + 3) ^ 3 + 
   d * (Complex.I + 3) ^ 2 + b * (Complex.I + 3) + a = 0) →
  (Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) →
  d.natAbs = 167 := by
sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_l3597_359751


namespace NUMINAMATH_CALUDE_iphone_sales_l3597_359788

theorem iphone_sales (iphone_price : ℝ) (ipad_count : ℕ) (ipad_price : ℝ)
                     (appletv_count : ℕ) (appletv_price : ℝ) (average_price : ℝ) :
  iphone_price = 1000 →
  ipad_count = 20 →
  ipad_price = 900 →
  appletv_count = 80 →
  appletv_price = 200 →
  average_price = 670 →
  ∃ (iphone_count : ℕ),
    (iphone_count : ℝ) * iphone_price + (ipad_count : ℝ) * ipad_price + (appletv_count : ℝ) * appletv_price =
    average_price * ((iphone_count : ℝ) + (ipad_count : ℝ) + (appletv_count : ℝ)) ∧
    iphone_count = 100 :=
by sorry

end NUMINAMATH_CALUDE_iphone_sales_l3597_359788


namespace NUMINAMATH_CALUDE_sally_cost_theorem_l3597_359779

def lightning_cost : ℝ := 140000

def mater_cost : ℝ := 0.1 * lightning_cost

def sally_cost : ℝ := 3 * mater_cost

theorem sally_cost_theorem : sally_cost = 42000 := by
  sorry

end NUMINAMATH_CALUDE_sally_cost_theorem_l3597_359779


namespace NUMINAMATH_CALUDE_ap_has_ten_terms_l3597_359794

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  n : ℕ                -- number of terms
  a : ℝ                -- first term
  d : ℝ                -- common difference
  n_even : Even n
  sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 28
  sum_even : (n / 2) * (2 * a + n * d) = 38
  last_first_diff : a + (n - 1) * d - a = 16

/-- Theorem stating that an arithmetic progression with the given properties has 10 terms -/
theorem ap_has_ten_terms (ap : ArithmeticProgression) : ap.n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ap_has_ten_terms_l3597_359794


namespace NUMINAMATH_CALUDE_min_overlap_brown_eyes_and_lunch_box_l3597_359774

theorem min_overlap_brown_eyes_and_lunch_box 
  (total_students : ℕ) 
  (brown_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 40) 
  (h2 : brown_eyes = 18) 
  (h3 : lunch_box = 25) : 
  ∃ (overlap : ℕ), 
    overlap ≥ brown_eyes + lunch_box - total_students ∧ 
    overlap = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_brown_eyes_and_lunch_box_l3597_359774


namespace NUMINAMATH_CALUDE_go_out_is_better_l3597_359777

/-- Represents the decision of the fishing boat -/
inductive Decision
| GoOut
| StayIn

/-- Represents the weather conditions -/
inductive Weather
| Good
| Bad

/-- The profit or loss for each scenario -/
def profit (d : Decision) (w : Weather) : ℝ :=
  match d, w with
  | Decision.GoOut, Weather.Good => 6000
  | Decision.GoOut, Weather.Bad => -8000
  | Decision.StayIn, _ => -1000

/-- The probability of each weather condition -/
def weather_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Good => 0.6
  | Weather.Bad => 0.4

/-- The expected value of a decision -/
def expected_value (d : Decision) : ℝ :=
  (profit d Weather.Good * weather_prob Weather.Good) +
  (profit d Weather.Bad * weather_prob Weather.Bad)

/-- Theorem stating that going out to sea has a higher expected value -/
theorem go_out_is_better :
  expected_value Decision.GoOut > expected_value Decision.StayIn :=
by sorry

end NUMINAMATH_CALUDE_go_out_is_better_l3597_359777


namespace NUMINAMATH_CALUDE_brad_siblings_product_l3597_359705

/-- A family structure with a focus on two siblings -/
structure Family :=
  (total_sisters : ℕ)
  (total_brothers : ℕ)
  (sarah_sisters : ℕ)
  (sarah_brothers : ℕ)

/-- The number of sisters and brothers that Brad has -/
def brad_siblings (f : Family) : ℕ × ℕ :=
  (f.total_sisters, f.total_brothers - 1)

/-- The theorem stating the product of Brad's siblings -/
theorem brad_siblings_product (f : Family) 
  (h1 : f.sarah_sisters = 4)
  (h2 : f.sarah_brothers = 7)
  (h3 : f.total_sisters = f.sarah_sisters + 1)
  (h4 : f.total_brothers = f.sarah_brothers + 1) :
  (brad_siblings f).1 * (brad_siblings f).2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_brad_siblings_product_l3597_359705


namespace NUMINAMATH_CALUDE_stating_sweet_apple_percentage_correct_l3597_359746

/-- Represents the percentage of sweet apples in Chang's Garden. -/
def sweet_apple_percentage : ℝ := 75

/-- Represents the total number of apples sold. -/
def total_apples : ℕ := 100

/-- Represents the price of a sweet apple in dollars. -/
def sweet_apple_price : ℝ := 0.5

/-- Represents the price of a sour apple in dollars. -/
def sour_apple_price : ℝ := 0.1

/-- Represents the total earnings from selling all apples in dollars. -/
def total_earnings : ℝ := 40

/-- 
Theorem stating that the percentage of sweet apples is correct given the conditions.
-/
theorem sweet_apple_percentage_correct : 
  sweet_apple_price * (sweet_apple_percentage / 100 * total_apples) + 
  sour_apple_price * ((100 - sweet_apple_percentage) / 100 * total_apples) = 
  total_earnings := by sorry

end NUMINAMATH_CALUDE_stating_sweet_apple_percentage_correct_l3597_359746


namespace NUMINAMATH_CALUDE_xiaoxiao_types_faster_l3597_359775

/-- Represents a typist with their typing data -/
structure Typist where
  name : String
  characters : ℕ
  minutes : ℕ

/-- Calculate the typing speed in characters per minute -/
def typingSpeed (t : Typist) : ℚ :=
  t.characters / t.minutes

/-- Determine if one typist is faster than another -/
def isFaster (t1 t2 : Typist) : Prop :=
  typingSpeed t1 > typingSpeed t2

theorem xiaoxiao_types_faster :
  let taoqi : Typist := { name := "淘气", characters := 200, minutes := 5 }
  let xiaoxiao : Typist := { name := "笑笑", characters := 132, minutes := 3 }
  isFaster xiaoxiao taoqi := by
  sorry

end NUMINAMATH_CALUDE_xiaoxiao_types_faster_l3597_359775


namespace NUMINAMATH_CALUDE_unique_divisibility_condition_l3597_359717

theorem unique_divisibility_condition (n : ℕ) : n > 1 → (
  (∃! a : ℕ, 0 < a ∧ a ≤ Nat.factorial n ∧ (Nat.factorial n ∣ a^n + 1)) ↔ n = 2
) := by sorry

end NUMINAMATH_CALUDE_unique_divisibility_condition_l3597_359717


namespace NUMINAMATH_CALUDE_sewing_time_proof_l3597_359756

/-- The time it takes to sew one dress -/
def time_per_dress (num_dresses : ℕ) (weekly_sewing_time : ℕ) (total_weeks : ℕ) : ℕ :=
  (weekly_sewing_time * total_weeks) / num_dresses

/-- Theorem stating that the time to sew one dress is 12 hours -/
theorem sewing_time_proof (num_dresses : ℕ) (weekly_sewing_time : ℕ) (total_weeks : ℕ) 
  (h1 : num_dresses = 5)
  (h2 : weekly_sewing_time = 4)
  (h3 : total_weeks = 15) :
  time_per_dress num_dresses weekly_sewing_time total_weeks = 12 := by
  sorry

end NUMINAMATH_CALUDE_sewing_time_proof_l3597_359756


namespace NUMINAMATH_CALUDE_projectile_trajectory_area_l3597_359747

open Real

/-- The area enclosed by the locus of highest points of projectile trajectories. -/
theorem projectile_trajectory_area (v g : ℝ) (h : ℝ := v^2 / (8 * g)) : 
  ∃ (area : ℝ), area = (3 * π / 32) * (v^4 / g^2) :=
by sorry

end NUMINAMATH_CALUDE_projectile_trajectory_area_l3597_359747


namespace NUMINAMATH_CALUDE_remaining_rolls_to_sell_l3597_359750

/-- Calculates the remaining rolls of gift wrap Nellie needs to sell -/
theorem remaining_rolls_to_sell 
  (total_rolls : ℕ) 
  (sold_to_grandmother : ℕ) 
  (sold_to_uncle : ℕ) 
  (sold_to_neighbor : ℕ) 
  (h1 : total_rolls = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_rolls - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end NUMINAMATH_CALUDE_remaining_rolls_to_sell_l3597_359750


namespace NUMINAMATH_CALUDE_max_rearrangeable_guests_correct_l3597_359781

/-- Represents a hotel with rooms numbered from 101 to 200 --/
structure Hotel :=
  (rooms : Finset Nat)
  (room_capacity : Nat → Nat)
  (room_range : ∀ r ∈ rooms, 101 ≤ r ∧ r ≤ 200)
  (capacity_matches_number : ∀ r ∈ rooms, room_capacity r = r)

/-- The maximum number of guests that can always be rearranged --/
def max_rearrangeable_guests (h : Hotel) : Nat :=
  8824

/-- Theorem stating that max_rearrangeable_guests is correct --/
theorem max_rearrangeable_guests_correct (h : Hotel) :
  ∀ n : Nat, n ≤ max_rearrangeable_guests h →
  (∀ vacated : h.rooms, ∃ destination : h.rooms,
    vacated ≠ destination ∧
    h.room_capacity destination ≥ h.room_capacity vacated) :=
sorry

#check max_rearrangeable_guests_correct

end NUMINAMATH_CALUDE_max_rearrangeable_guests_correct_l3597_359781


namespace NUMINAMATH_CALUDE_expression_evaluation_l3597_359754

theorem expression_evaluation : 
  (3 - 4 * (5 - 6)⁻¹)⁻¹ * (1 - 2⁻¹) = (1 : ℚ) / 14 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3597_359754


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l3597_359745

/-- A three-digit positive integer -/
def ThreeDigitInt := {n : ℕ // 100 ≤ n ∧ n < 1000}

/-- Extracts digits from a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec extract (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else extract (m / 10) ((m % 10) :: acc)
  extract n []

/-- Sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := (digits n).sum

/-- All digits in two numbers are different -/
def allDigitsDifferent (a b : ThreeDigitInt) : Prop :=
  (digits a.val ++ digits b.val).Nodup

theorem smallest_digit_sum_of_sum (a b : ThreeDigitInt) 
  (h : allDigitsDifferent a b) : 
  ∃ (S : ℕ), S = a.val + b.val ∧ 1000 ≤ S ∧ S < 10000 ∧ 
  (∀ (T : ℕ), T = a.val + b.val → 1000 ≤ T ∧ T < 10000 → digitSum S ≤ digitSum T) ∧
  digitSum S = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l3597_359745


namespace NUMINAMATH_CALUDE_minuend_subtrahend_difference_problem_l3597_359704

theorem minuend_subtrahend_difference_problem :
  ∃ (a b c : ℤ),
    (a + b + c = 1024) ∧
    (c = b - 88) ∧
    (a = b + c) ∧
    (a = 712) ∧
    (b = 400) ∧
    (c = 312) := by
  sorry

end NUMINAMATH_CALUDE_minuend_subtrahend_difference_problem_l3597_359704


namespace NUMINAMATH_CALUDE_f_minus_g_at_7_l3597_359702

def f : ℝ → ℝ := fun _ ↦ 3

def g : ℝ → ℝ := fun _ ↦ 5

theorem f_minus_g_at_7 : f 7 - g 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_g_at_7_l3597_359702


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l3597_359723

/-- Given a quadratic expression 3x^2 + 9x + 20, prove that when written in the form a(x - h)^2 + k, the value of h is -3/2. -/
theorem quadratic_vertex_form_h (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l3597_359723


namespace NUMINAMATH_CALUDE_new_person_weight_l3597_359726

theorem new_person_weight (initial_count : ℕ) (initial_weight : ℝ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  weight_increase = 6 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3597_359726


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3597_359759

theorem decimal_sum_to_fraction : 
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3597_359759


namespace NUMINAMATH_CALUDE_parallelogram_properties_l3597_359727

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  opposite : v1 = v2

/-- The fourth vertex and diagonal intersection of a specific parallelogram -/
theorem parallelogram_properties (p : Parallelogram) 
  (h1 : p.v1 = (2, -3))
  (h2 : p.v2 = (8, 5))
  (h3 : p.v3 = (5, 0)) :
  p.v4 = (5, 2) ∧ 
  (((p.v1.1 + p.v2.1) / 2, (p.v1.2 + p.v2.2) / 2) : ℝ × ℝ) = (5, 1) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l3597_359727


namespace NUMINAMATH_CALUDE_train_length_l3597_359729

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 8 → ∃ length : ℝ, 
  (length ≥ 133.36 ∧ length ≤ 133.37) ∧ length = speed * time * (1000 / 3600) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3597_359729


namespace NUMINAMATH_CALUDE_derivative_sum_at_points_l3597_359711

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + 2*x + 5

-- Define the derivative of f
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + 2

-- Theorem statement
theorem derivative_sum_at_points (m : ℝ) : f' m 2 + f' m (-2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sum_at_points_l3597_359711


namespace NUMINAMATH_CALUDE_adam_received_one_smiley_l3597_359700

/-- Represents the number of smileys each friend received -/
structure SmileyCounts where
  adam : ℕ
  mojmir : ℕ
  petr : ℕ
  pavel : ℕ

/-- The conditions of the problem -/
def validSmileyCounts (counts : SmileyCounts) : Prop :=
  counts.adam + counts.mojmir + counts.petr + counts.pavel = 52 ∧
  counts.adam ≥ 1 ∧
  counts.mojmir ≥ 1 ∧
  counts.petr ≥ 1 ∧
  counts.pavel ≥ 1 ∧
  counts.petr + counts.pavel = 33 ∧
  counts.mojmir > counts.adam ∧
  counts.mojmir > counts.petr ∧
  counts.mojmir > counts.pavel

theorem adam_received_one_smiley (counts : SmileyCounts) 
  (h : validSmileyCounts counts) : counts.adam = 1 := by
  sorry

end NUMINAMATH_CALUDE_adam_received_one_smiley_l3597_359700


namespace NUMINAMATH_CALUDE_five_chairs_cost_l3597_359739

/-- The cost of a single plastic chair -/
def chair_cost : ℝ := sorry

/-- The cost of a portable table -/
def table_cost : ℝ := sorry

/-- Three chairs cost the same as one table -/
axiom chair_table_relation : 3 * chair_cost = table_cost

/-- One table and two chairs cost $55 -/
axiom total_cost : table_cost + 2 * chair_cost = 55

/-- The cost of five plastic chairs is $55 -/
theorem five_chairs_cost : 5 * chair_cost = 55 := by sorry

end NUMINAMATH_CALUDE_five_chairs_cost_l3597_359739


namespace NUMINAMATH_CALUDE_beanie_babies_total_l3597_359725

/-- The total number of beanie babies owned by Lori and Sydney -/
def total_beanie_babies (lori_beanie_babies : ℕ) (lori_sydney_ratio : ℕ) : ℕ :=
  lori_beanie_babies + (lori_beanie_babies / lori_sydney_ratio)

/-- Theorem: Given Lori has 300 beanie babies and owns 15 times as many as Sydney,
    the total number of beanie babies they have is 320. -/
theorem beanie_babies_total :
  total_beanie_babies 300 15 = 320 := by
  sorry

end NUMINAMATH_CALUDE_beanie_babies_total_l3597_359725


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3597_359710

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a > 0} = Set.Ioi (1/2 : ℝ) ∪ Set.Iic (-1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3597_359710


namespace NUMINAMATH_CALUDE_travel_agency_comparison_l3597_359791

/-- Calculates the total cost for Travel Agency A -/
def costA (fullPrice : ℕ) (numStudents : ℕ) : ℕ :=
  fullPrice + numStudents * (fullPrice / 2)

/-- Calculates the total cost for Travel Agency B -/
def costB (fullPrice : ℕ) (numPeople : ℕ) : ℕ :=
  numPeople * (fullPrice * 60 / 100)

theorem travel_agency_comparison (fullPrice : ℕ) :
  (fullPrice = 240) →
  (costA fullPrice 5 < costB fullPrice 6) ∧
  (costB fullPrice 3 < costA fullPrice 2) := by
  sorry


end NUMINAMATH_CALUDE_travel_agency_comparison_l3597_359791


namespace NUMINAMATH_CALUDE_circle_condition_l3597_359789

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0

-- Define what it means for an equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  a^2 = a + 2 ∧ a ≠ 0

-- Theorem statement
theorem circle_condition (a : ℝ) :
  represents_circle a ↔ a = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l3597_359789


namespace NUMINAMATH_CALUDE_sum_of_divisors_theorem_l3597_359755

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 3780, then i + j + k = 8 -/
theorem sum_of_divisors_theorem (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 3780 → i + j + k = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_theorem_l3597_359755


namespace NUMINAMATH_CALUDE_exists_minimum_top_number_l3597_359709

/-- Represents a square pyramid of blocks -/
structure SquarePyramid where
  base : Matrix (Fin 4) (Fin 4) ℕ
  layer2 : Matrix (Fin 3) (Fin 3) ℕ
  layer3 : Matrix (Fin 2) (Fin 2) ℕ
  top : ℕ

/-- Checks if the pyramid is valid according to the given conditions -/
def isValidPyramid (p : SquarePyramid) : Prop :=
  (∀ i j, p.base i j ∈ Finset.range 17) ∧
  (∀ i j, p.layer2 i j = p.base (i+1) (j+1) + p.base (i+1) j + p.base i (j+1)) ∧
  (∀ i j, p.layer3 i j = p.layer2 (i+1) (j+1) + p.layer2 (i+1) j + p.layer2 i (j+1)) ∧
  (p.top = p.layer3 1 1 + p.layer3 1 0 + p.layer3 0 1)

/-- The main theorem statement -/
theorem exists_minimum_top_number :
  ∃ (min : ℕ), ∀ (p : SquarePyramid), isValidPyramid p → p.top ≥ min :=
sorry


end NUMINAMATH_CALUDE_exists_minimum_top_number_l3597_359709


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l3597_359795

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 9 * d)
  (h_geom_mean : a k ^ 2 = a 1 * a (2 * k)) :
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l3597_359795


namespace NUMINAMATH_CALUDE_class_composition_l3597_359706

theorem class_composition (boys_score girls_score class_average : ℝ) 
  (boys_score_val : boys_score = 80)
  (girls_score_val : girls_score = 90)
  (class_average_val : class_average = 86) :
  let boys_percentage : ℝ := 40
  let girls_percentage : ℝ := 100 - boys_percentage
  class_average = (boys_percentage * boys_score + girls_percentage * girls_score) / 100 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l3597_359706


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3597_359778

/-- A quadratic function with vertex (2, 5) passing through (0, 0) has a = -5/4 --/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- quadratic function definition
  (2, 5) = (2, a * 2^2 + b * 2 + c) →     -- vertex condition
  (0, 0) = (0, a * 0^2 + b * 0 + c) →     -- point condition
  a = -5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3597_359778


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3597_359730

/-- Define the diamond operation -/
def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

/-- Theorem: If A ◇ 5 = 82, then A = 12 -/
theorem diamond_equation_solution :
  ∀ A : ℝ, diamond A 5 = 82 → A = 12 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3597_359730


namespace NUMINAMATH_CALUDE_not_prime_p_l3597_359724

theorem not_prime_p (x k : ℕ) (p : ℕ) (h : x^5 + 2*x + 3 = p*k) : ¬ Nat.Prime p := by
  sorry

end NUMINAMATH_CALUDE_not_prime_p_l3597_359724


namespace NUMINAMATH_CALUDE_profit_growth_rate_l3597_359790

theorem profit_growth_rate (initial_profit target_profit : ℝ) (growth_rate : ℝ) (months : ℕ) :
  initial_profit * (1 + growth_rate / 100) ^ months = target_profit →
  growth_rate = 25 :=
by
  intro h
  -- Proof goes here
  sorry

#check profit_growth_rate 1.6 2.5 25 2

end NUMINAMATH_CALUDE_profit_growth_rate_l3597_359790


namespace NUMINAMATH_CALUDE_difference_calculation_l3597_359796

theorem difference_calculation (total : ℝ) (h : total = 8000) : 
  (1 / 10 : ℝ) * total - (1 / 20 : ℝ) * (1 / 100 : ℝ) * total = 796 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l3597_359796


namespace NUMINAMATH_CALUDE_exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5_l3597_359713

theorem exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5 :
  ∃ n : ℕ+, 24 ∣ n ∧ 9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.5 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5_l3597_359713


namespace NUMINAMATH_CALUDE_player_B_winning_condition_l3597_359721

/-- Represents the game state -/
structure GameState where
  stones : ℕ

/-- Represents a player's move -/
structure Move where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  move.pile1 > 0 ∧ move.pile2 > 0 ∧ move.pile3 > 0 ∧
  move.pile1 + move.pile2 + move.pile3 = state.stones ∧
  (move.pile1 > move.pile2 ∧ move.pile1 > move.pile3) ∨
  (move.pile2 > move.pile1 ∧ move.pile2 > move.pile3) ∨
  (move.pile3 > move.pile1 ∧ move.pile3 > move.pile2)

/-- Defines a winning strategy for Player B -/
def player_B_has_winning_strategy (n : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (initial_move : Move),
      is_valid_move { stones := n } initial_move →
        ∃ (game_sequence : ℕ → GameState),
          game_sequence 0 = { stones := n } ∧
          (∀ i : ℕ, is_valid_move (game_sequence i) (strategy (game_sequence i))) ∧
          ∃ (end_state : ℕ), ¬is_valid_move (game_sequence end_state) (strategy (game_sequence end_state))

/-- The main theorem stating the condition for Player B's winning strategy -/
theorem player_B_winning_condition {a b : ℕ} (ha : a > 1) (hb : b > 1) :
  player_B_has_winning_strategy (a^b) ↔ ∃ k : ℕ, k > 1 ∧ (a^b = 3^k ∨ a^b = 3^k - 1) :=
sorry

end NUMINAMATH_CALUDE_player_B_winning_condition_l3597_359721


namespace NUMINAMATH_CALUDE_amanda_hiking_trip_l3597_359712

/-- Represents Amanda's hiking trip -/
def hiking_trip (total_distance : ℚ) : Prop :=
  let first_segment := total_distance / 4
  let forest_segment := 25
  let mountain_segment := total_distance / 6
  let plain_segment := 2 * forest_segment
  first_segment + forest_segment + mountain_segment + plain_segment = total_distance

theorem amanda_hiking_trip :
  ∃ (total_distance : ℚ), hiking_trip total_distance ∧ total_distance = 900 / 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_hiking_trip_l3597_359712


namespace NUMINAMATH_CALUDE_frog_arrangements_count_l3597_359733

/-- The number of ways to arrange 8 distinguishable frogs (3 green, 4 red, 1 blue) in a row,
    such that no two frogs of the same color are adjacent. -/
def frog_arrangements : ℕ :=
  8 * (Nat.choose 7 3) * (Nat.factorial 3) * (Nat.factorial 4)

/-- Theorem stating that the number of valid frog arrangements is 40320. -/
theorem frog_arrangements_count : frog_arrangements = 40320 := by
  sorry

end NUMINAMATH_CALUDE_frog_arrangements_count_l3597_359733


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3597_359714

theorem quadratic_roots_sum (k₁ k₂ : ℝ) : 
  36 * k₁^2 - 200 * k₁ + 49 = 0 →
  36 * k₂^2 - 200 * k₂ + 49 = 0 →
  k₁ / k₂ + k₂ / k₁ = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3597_359714


namespace NUMINAMATH_CALUDE_y_over_x_bounds_y_minus_x_bounds_x_squared_plus_y_squared_bounds_l3597_359772

-- Define the condition
def satisfies_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 1 = 0

-- Theorem for the maximum and minimum values of y/x
theorem y_over_x_bounds {x y : ℝ} (h : satisfies_equation x y) (hx : x ≠ 0) :
  y / x ≤ Real.sqrt 3 ∧ y / x ≥ -Real.sqrt 3 :=
sorry

-- Theorem for the maximum and minimum values of y - x
theorem y_minus_x_bounds {x y : ℝ} (h : satisfies_equation x y) :
  y - x ≤ -2 + Real.sqrt 6 ∧ y - x ≥ -2 - Real.sqrt 6 :=
sorry

-- Theorem for the maximum and minimum values of x^2 + y^2
theorem x_squared_plus_y_squared_bounds {x y : ℝ} (h : satisfies_equation x y) :
  x^2 + y^2 ≤ 7 + 4 * Real.sqrt 3 ∧ x^2 + y^2 ≥ 7 - 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_y_over_x_bounds_y_minus_x_bounds_x_squared_plus_y_squared_bounds_l3597_359772


namespace NUMINAMATH_CALUDE_no_valid_x_for_mean_12_l3597_359782

theorem no_valid_x_for_mean_12 : 
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 1917 + 2114 + x) / 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_x_for_mean_12_l3597_359782


namespace NUMINAMATH_CALUDE_certain_number_problem_l3597_359784

theorem certain_number_problem (x : ℝ) : ((2 * (x + 5)) / 5) - 5 = 22 → x = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3597_359784


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3597_359719

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3597_359719


namespace NUMINAMATH_CALUDE_no_obtuse_angle_at_center_l3597_359737

/-- Represents a point on a circle -/
structure CirclePoint where
  arc : Fin 3
  position : ℝ
  h_position : 0 ≤ position ∧ position < 2 * Real.pi / 3

/-- Represents a configuration of 6 points on a circle -/
def CircleConfiguration := Fin 6 → CirclePoint

/-- Checks if three points form an obtuse angle at the center -/
def has_obtuse_angle_at_center (config : CircleConfiguration) (p1 p2 p3 : Fin 6) : Prop :=
  ∃ θ, θ > Real.pi / 2 ∧
    θ = min (2 * Real.pi / 3) (abs ((config p2).position - (config p1).position) +
      abs ((config p3).position - (config p2).position) +
      abs ((config p1).position - (config p3).position))

/-- The main theorem statement -/
theorem no_obtuse_angle_at_center (config : CircleConfiguration) :
  ∀ p1 p2 p3 : Fin 6, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  ¬(has_obtuse_angle_at_center config p1 p2 p3) :=
sorry

end NUMINAMATH_CALUDE_no_obtuse_angle_at_center_l3597_359737


namespace NUMINAMATH_CALUDE_swanson_class_avg_l3597_359762

/-- The average number of zits per kid in Ms. Swanson's class -/
def swanson_avg : ℝ := 5

/-- The number of kids in Ms. Swanson's class -/
def swanson_kids : ℕ := 25

/-- The number of kids in Mr. Jones' class -/
def jones_kids : ℕ := 32

/-- The average number of zits per kid in Mr. Jones' class -/
def jones_avg : ℝ := 6

/-- The difference in total zits between Mr. Jones' and Ms. Swanson's classes -/
def zit_difference : ℕ := 67

theorem swanson_class_avg : 
  swanson_avg * swanson_kids + zit_difference = jones_avg * jones_kids := by
  sorry

#check swanson_class_avg

end NUMINAMATH_CALUDE_swanson_class_avg_l3597_359762


namespace NUMINAMATH_CALUDE_largest_degree_with_asymptote_l3597_359792

-- Define the denominator of our rational function
def q (x : ℝ) : ℝ := 3 * x^6 + 2 * x^3 - x + 4

-- Define a proposition that checks if a polynomial has a horizontal asymptote when divided by q(x)
def has_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x > M, |p x / q x - L| < ε

-- Define a function to get the degree of a polynomial
noncomputable def poly_degree (p : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem largest_degree_with_asymptote :
  ∃ (p : ℝ → ℝ), poly_degree p = 6 ∧ has_horizontal_asymptote p ∧
  ∀ (p' : ℝ → ℝ), poly_degree p' > 6 → ¬(has_horizontal_asymptote p') :=
sorry

end NUMINAMATH_CALUDE_largest_degree_with_asymptote_l3597_359792


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3597_359748

def set_A (a : ℝ) : Set ℝ := {-2, 3*a-1, a^2-3}
def set_B (a : ℝ) : Set ℝ := {a-2, a-1, a+1}

theorem intersection_implies_a_value (a : ℝ) :
  set_A a ∩ set_B a = {-2} → a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3597_359748


namespace NUMINAMATH_CALUDE_prime_fraction_sum_of_reciprocals_l3597_359757

theorem prime_fraction_sum_of_reciprocals (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ (m : ℕ) (x y : ℕ+), 3 ≤ m ∧ m ≤ p - 2 ∧ (m : ℚ) / (p^2 : ℚ) = (1 : ℚ) / (x : ℚ) + (1 : ℚ) / (y : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prime_fraction_sum_of_reciprocals_l3597_359757


namespace NUMINAMATH_CALUDE_sibling_discount_calculation_l3597_359753

/-- Represents the tuition cost at the music school -/
def regular_tuition : ℕ := 45

/-- Represents the discounted cost for both children -/
def discounted_cost : ℕ := 75

/-- Represents the number of children -/
def num_children : ℕ := 2

/-- Calculates the sibling discount -/
def sibling_discount : ℕ :=
  regular_tuition * num_children - discounted_cost

theorem sibling_discount_calculation :
  sibling_discount = 15 := by sorry

end NUMINAMATH_CALUDE_sibling_discount_calculation_l3597_359753


namespace NUMINAMATH_CALUDE_guppies_per_day_l3597_359770

/-- The number of guppies Jason's moray eel eats per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish Jason has -/
def num_betta_fish : ℕ := 5

/-- The number of guppies each betta fish eats per day -/
def betta_fish_guppies : ℕ := 7

/-- Theorem: Jason needs to buy 55 guppies per day -/
theorem guppies_per_day : 
  moray_eel_guppies + num_betta_fish * betta_fish_guppies = 55 := by
  sorry

end NUMINAMATH_CALUDE_guppies_per_day_l3597_359770


namespace NUMINAMATH_CALUDE_distance_traveled_l3597_359769

/-- Given a speed of 75 km/hr and a time of 4 hours, prove that the distance traveled is 300 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 75) (h2 : time = 4) :
  speed * time = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3597_359769


namespace NUMINAMATH_CALUDE_multiples_2_3_not_5_l3597_359771

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n)

theorem multiples_2_3_not_5 (max : ℕ) (h : max = 200) :
  (count_multiples 2 max + count_multiples 3 max - count_multiples 6 max) -
  (count_multiples 10 max + count_multiples 15 max - count_multiples 30 max) = 107 :=
by sorry

end NUMINAMATH_CALUDE_multiples_2_3_not_5_l3597_359771


namespace NUMINAMATH_CALUDE_nine_digit_sum_l3597_359752

-- Define the type for digits 1-9
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

-- Define the structure for the nine-digit number
structure NineDigitNumber where
  digits : Fin 9 → Digit
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

-- Define the property that each two-digit segment is a product of two single-digit numbers
def validSegments (n : NineDigitNumber) : Prop :=
  ∀ i : Fin 8, ∃ (x y : Digit), 
    (n.digits i).val * 10 + (n.digits (i + 1)).val = x.val * y.val

-- Define the function to calculate the sum of ABC + DEF + GHI
def sumSegments (n : NineDigitNumber) : ℕ :=
  ((n.digits 0).val * 100 + (n.digits 1).val * 10 + (n.digits 2).val) +
  ((n.digits 3).val * 100 + (n.digits 4).val * 10 + (n.digits 5).val) +
  ((n.digits 6).val * 100 + (n.digits 7).val * 10 + (n.digits 8).val)

-- State the theorem
theorem nine_digit_sum (n : NineDigitNumber) (h : validSegments n) : 
  sumSegments n = 1440 :=
sorry

end NUMINAMATH_CALUDE_nine_digit_sum_l3597_359752


namespace NUMINAMATH_CALUDE_income_calculation_l3597_359776

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Proves that given a person's income to expenditure ratio of 5:4 and savings of Rs. 3000, 
    the person's income is Rs. 15000. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) 
  (h1 : income_ratio = 5) 
  (h2 : expenditure_ratio = 4) 
  (h3 : savings = 3000) : 
  calculate_income income_ratio expenditure_ratio savings = 15000 := by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l3597_359776


namespace NUMINAMATH_CALUDE_tangent_product_30_60_l3597_359736

theorem tangent_product_30_60 (A B : Real) (hA : A = 30 * π / 180) (hB : B = 60 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = (3 + 4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_30_60_l3597_359736


namespace NUMINAMATH_CALUDE_range_of_H_l3597_359735

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ -4 ≤ y ∧ y ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l3597_359735


namespace NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l3597_359731

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point where a line intersects the x-axis -/
def XAxisIntersection : ℝ → ℝ × ℝ := λ x ↦ (x, 0)

/-- Theorem: Tangent line intersection for two specific circles -/
theorem tangent_intersection_x_coordinate :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (12, 0), radius := 5 }
  ∃ (x : ℝ), x > 0 ∧ 
    ∃ (l : Set (ℝ × ℝ)), 
      (XAxisIntersection x ∈ l) ∧ 
      (∃ p1 ∈ l, (p1.1 - c1.center.1)^2 + (p1.2 - c1.center.2)^2 = c1.radius^2) ∧
      (∃ p2 ∈ l, (p2.1 - c2.center.1)^2 + (p2.2 - c2.center.2)^2 = c2.radius^2) ∧
      x = 9/2 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l3597_359731


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l3597_359760

def cloth_length : ℕ := 80
def selling_price : ℕ := 10000
def profit_per_meter : ℕ := 7

theorem cost_price_per_meter (total_profit : ℕ) (cost_price : ℕ) :
  total_profit = cloth_length * profit_per_meter →
  selling_price = cost_price + total_profit →
  cost_price / cloth_length = 118 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l3597_359760


namespace NUMINAMATH_CALUDE_base7_product_sum_theorem_l3597_359742

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := sorry

theorem base7_product_sum_theorem :
  let a := 35
  let b := 21
  let product := multiplyBase7 a b
  let digitSum := sumDigitsBase7 product
  multiplyBase7 digitSum 3 = 63
  := by sorry

end NUMINAMATH_CALUDE_base7_product_sum_theorem_l3597_359742


namespace NUMINAMATH_CALUDE_f_equals_three_implies_x_is_sqrt_three_l3597_359793

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_equals_three_implies_x_is_sqrt_three :
  ∀ x : ℝ, f x = 3 → x = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_f_equals_three_implies_x_is_sqrt_three_l3597_359793


namespace NUMINAMATH_CALUDE_min_distance_on_feb_9th_l3597_359738

/-- Represents the squared distance between a space probe and Mars as a function of time -/
def D (a b c : ℝ) (t : ℝ) : ℝ := a * t^2 + b * t + c

/-- Theorem stating that the minimum distance occurs on February 9th -/
theorem min_distance_on_feb_9th (a b c : ℝ) :
  D a b c (-9) = 25 →
  D a b c 0 = 4 →
  D a b c 3 = 9 →
  ∃ (t_min : ℝ), t_min = -1 ∧ ∀ (t : ℝ), D a b c t_min ≤ D a b c t :=
sorry

end NUMINAMATH_CALUDE_min_distance_on_feb_9th_l3597_359738


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l3597_359763

def is_divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, (945 * 10000 + n * 1000 + 631) = 11 * k

theorem seven_digit_divisible_by_11 (n : ℕ) (h : n < 10) :
  is_divisible_by_11 n → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l3597_359763


namespace NUMINAMATH_CALUDE_decreasing_function_range_l3597_359749

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → f x > f y) →
  (1 - a ∈ Set.Ioo (-1) 1) →
  (a^2 - 1 ∈ Set.Ioo (-1) 1) →
  f (1 - a) < f (a^2 - 1) →
  0 < a ∧ a < Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_decreasing_function_range_l3597_359749


namespace NUMINAMATH_CALUDE_cube_cutting_l3597_359766

theorem cube_cutting (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a^3 : ℕ) = 98 + b^3 → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l3597_359766


namespace NUMINAMATH_CALUDE_sequence_formula_l3597_359741

theorem sequence_formula (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l3597_359741


namespace NUMINAMATH_CALUDE_f_max_value_l3597_359786

def f (a b c : Real) : Real :=
  a * (1 - a + a * b) * (1 - a * b + a * b * c) * (1 - c)

theorem f_max_value (a b c : Real) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  f a b c ≤ 8/27 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l3597_359786


namespace NUMINAMATH_CALUDE_triangle_inequality_l3597_359703

/-- Given a triangle with sides a, b, c and area S, 
    the sum of squares of the sides is greater than or equal to 
    4 times the area multiplied by the square root of 3. 
    Equality holds if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S > 0)
  (h_S : S = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h_s : s = (a + b + c) / 2) : 
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 ∧ 
  (a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3597_359703


namespace NUMINAMATH_CALUDE_cot_15_plus_tan_45_l3597_359773

theorem cot_15_plus_tan_45 : Real.cos (15 * π / 180) / Real.sin (15 * π / 180) + Real.tan (45 * π / 180) = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_15_plus_tan_45_l3597_359773


namespace NUMINAMATH_CALUDE_complex_circle_equation_l3597_359722

/-- The set of complex numbers z satisfying |z-i| = |3-4i| forms a circle in the complex plane -/
theorem complex_circle_equation : 
  ∃ (center : ℂ) (radius : ℝ), 
    {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 - 4 * Complex.I)} = 
    {z : ℂ | Complex.abs (z - center) = radius} :=
sorry

end NUMINAMATH_CALUDE_complex_circle_equation_l3597_359722


namespace NUMINAMATH_CALUDE_wilson_sledding_l3597_359798

theorem wilson_sledding (tall_hills small_hills tall_runs small_runs : ℕ) 
  (h1 : tall_hills = 2)
  (h2 : small_hills = 3)
  (h3 : tall_runs = 4)
  (h4 : small_runs = tall_runs / 2)
  : tall_hills * tall_runs + small_hills * small_runs = 14 := by
  sorry

end NUMINAMATH_CALUDE_wilson_sledding_l3597_359798


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3597_359718

theorem inheritance_calculation (x : ℝ) : 
  x > 0 →
  (0.25 * x + 0.15 * (x - 0.25 * x) = 18000) →
  x = 50000 := by
sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3597_359718


namespace NUMINAMATH_CALUDE_sugar_price_proof_l3597_359707

/-- Proves that given the initial price of sugar as 6 Rs/kg, a new price of 7.50 Rs/kg, 
    and a reduction in consumption of 19.999999999999996%, the initial price of sugar is 6 Rs/kg. -/
theorem sugar_price_proof (initial_price : ℝ) (new_price : ℝ) (consumption_reduction : ℝ) : 
  initial_price = 6 ∧ new_price = 7.5 ∧ consumption_reduction = 19.999999999999996 → initial_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_proof_l3597_359707


namespace NUMINAMATH_CALUDE_chemistry_class_size_l3597_359783

theorem chemistry_class_size 
  (total_students : ℕ) 
  (chem_only : ℕ) 
  (bio_only : ℕ) 
  (both : ℕ) 
  (h1 : total_students = 70)
  (h2 : total_students = chem_only + bio_only + both)
  (h3 : chem_only + both = 2 * (bio_only + both))
  (h4 : both = 8) : 
  chem_only + both = 52 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l3597_359783


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3597_359744

theorem quadratic_max_value (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 15 * x + 9
  ∃ (max : ℝ), max = (111 : ℝ) / 4 ∧ ∀ y, f y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3597_359744


namespace NUMINAMATH_CALUDE_max_value_expression_l3597_359768

theorem max_value_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 10) :
  Real.sqrt (2 * x + 20) + Real.sqrt (26 - 2 * x) + Real.sqrt (3 * x) ≤ 4 * Real.sqrt 79 ∧
  (x = 10 → Real.sqrt (2 * x + 20) + Real.sqrt (26 - 2 * x) + Real.sqrt (3 * x) = 4 * Real.sqrt 79) := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3597_359768


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3597_359765

/-- Given a hyperbola with equation x²/121 - y²/81 = 1, 
    prove that the positive value n in its asymptote equations y = ±nx is 9/11 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  (x^2 / 121 - y^2 / 81 = 1) →
  (∃ (n : ℝ), n > 0 ∧ (y = n*x ∨ y = -n*x) ∧ n = 9/11) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3597_359765


namespace NUMINAMATH_CALUDE_units_digit_of_A_is_one_l3597_359767

-- Define the function for the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the expression for A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Theorem statement
theorem units_digit_of_A_is_one : unitsDigit A = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_A_is_one_l3597_359767


namespace NUMINAMATH_CALUDE_bob_has_winning_strategy_l3597_359720

/-- Represents a cell in the grid -/
structure Cell :=
  (row : ℕ)
  (col : ℕ)
  (value : ℚ)

/-- Represents the game state -/
structure GameState :=
  (grid : List (List Cell))
  (current_player : Bool)  -- true for Alice, false for Bob

/-- Checks if a cell is part of a continuous path from top to bottom -/
def is_part_of_path (grid : List (List Cell)) (cell : Cell) : Prop :=
  sorry

/-- Determines if there exists a winning path for Alice -/
def exists_winning_path (state : GameState) : Prop :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Cell

/-- Determines if a strategy is winning for Bob -/
def is_winning_strategy_for_bob (strategy : Strategy) : Prop :=
  ∀ (state : GameState), 
    (state.current_player = false) → 
    ¬(exists_winning_path (state))

/-- The main theorem stating that Bob has a winning strategy -/
theorem bob_has_winning_strategy : 
  ∃ (strategy : Strategy), is_winning_strategy_for_bob strategy :=
sorry

end NUMINAMATH_CALUDE_bob_has_winning_strategy_l3597_359720


namespace NUMINAMATH_CALUDE_roots_are_irrational_l3597_359780

/-- Given a real number k, this function represents the quadratic equation x^2 - 3kx + 2k^2 - 1 = 0 --/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - 3*k*x + 2*k^2 - 1 = 0

/-- The product of the roots of the quadratic equation is 7 --/
axiom root_product (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁ * x₂ = 7

/-- Definition of an irrational number --/
def is_irrational (x : ℝ) : Prop :=
  ∀ p q : ℤ, q ≠ 0 → x ≠ p / q

/-- The main theorem: the roots of the quadratic equation are irrational --/
theorem roots_are_irrational (k : ℝ) :
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ 
             is_irrational x₁ ∧ is_irrational x₂ :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l3597_359780
