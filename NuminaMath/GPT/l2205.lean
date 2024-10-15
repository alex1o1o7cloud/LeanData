import Mathlib

namespace NUMINAMATH_GPT_cube_distance_l2205_220523

-- The Lean 4 statement
theorem cube_distance (side_length : ℝ) (h1 h2 h3 : ℝ) (r s t : ℕ) 
  (h1_eq : h1 = 18) (h2_eq : h2 = 20) (h3_eq : h3 = 22) (side_length_eq : side_length = 15) :
  r = 57 ∧ s = 597 ∧ t = 3 ∧ r + s + t = 657 :=
by
  sorry

end NUMINAMATH_GPT_cube_distance_l2205_220523


namespace NUMINAMATH_GPT_determine_parabola_equation_l2205_220524

-- Define the conditions
def focus_on_line (focus : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, focus = (k - 2, k / 2 - 1)

-- Define the result equations
def is_standard_equation (eq : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, eq x y → x^2 = 4 * y) ∨ (∀ x y : ℝ, eq x y → y^2 = -8 * x)

-- Define the theorem stating that given the condition,
-- the standard equation is one of the two forms
theorem determine_parabola_equation (focus : ℝ × ℝ) (H : focus_on_line focus) :
  ∃ eq : ℝ → ℝ → Prop, is_standard_equation eq :=
sorry

end NUMINAMATH_GPT_determine_parabola_equation_l2205_220524


namespace NUMINAMATH_GPT_age_difference_of_siblings_l2205_220564

theorem age_difference_of_siblings (x : ℝ) 
  (h1 : 19 * x + 20 = 230) :
  |4 * x - 3 * x| = 210 / 19 := by
    sorry

end NUMINAMATH_GPT_age_difference_of_siblings_l2205_220564


namespace NUMINAMATH_GPT_problem1_problem2_l2205_220551

theorem problem1 : 2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  -- Proof omitted
  sorry

theorem problem2 : (Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2205_220551


namespace NUMINAMATH_GPT_sasha_kolya_distance_l2205_220508

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end NUMINAMATH_GPT_sasha_kolya_distance_l2205_220508


namespace NUMINAMATH_GPT_total_cups_needed_l2205_220582

def servings : Float := 18.0
def cups_per_serving : Float := 2.0

theorem total_cups_needed : servings * cups_per_serving = 36.0 :=
by
  sorry

end NUMINAMATH_GPT_total_cups_needed_l2205_220582


namespace NUMINAMATH_GPT_proposition_only_A_l2205_220503

def is_proposition (statement : String) : Prop := sorry

def statement_A : String := "Red beans grow in the southern country"
def statement_B : String := "They sprout several branches in spring"
def statement_C : String := "I hope you pick more"
def statement_D : String := "For these beans symbolize longing"

theorem proposition_only_A :
  is_proposition statement_A ∧
  ¬is_proposition statement_B ∧
  ¬is_proposition statement_C ∧
  ¬is_proposition statement_D := 
sorry

end NUMINAMATH_GPT_proposition_only_A_l2205_220503


namespace NUMINAMATH_GPT_dot_product_v_w_l2205_220567

def v : ℝ × ℝ := (-5, 3)
def w : ℝ × ℝ := (7, -9)

theorem dot_product_v_w : v.1 * w.1 + v.2 * w.2 = -62 := 
  by sorry

end NUMINAMATH_GPT_dot_product_v_w_l2205_220567


namespace NUMINAMATH_GPT_no_A_with_integer_roots_l2205_220566

theorem no_A_with_integer_roots 
  (A : ℕ) 
  (h1 : A > 0) 
  (h2 : A < 10) 
  : ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ p + q = 10 + A ∧ p * q = 10 * A + A :=
by sorry

end NUMINAMATH_GPT_no_A_with_integer_roots_l2205_220566


namespace NUMINAMATH_GPT_arithmetic_sequence_l2205_220517

variable (a : ℕ → ℕ)
variable (h : a 1 + 3 * a 8 + a 15 = 120)

theorem arithmetic_sequence (h : a 1 + 3 * a 8 + a 15 = 120) : a 2 + a 14 = 48 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_l2205_220517


namespace NUMINAMATH_GPT_count_four_digit_integers_with_conditions_l2205_220578

def is_four_digit_integer (n : Nat) : Prop := 1000 ≤ n ∧ n < 10000

def thousands_digit_is_seven (n : Nat) : Prop := 
  (n / 1000) % 10 = 7

def hundreds_digit_is_odd (n : Nat) : Prop := 
  let hd := (n / 100) % 10
  hd % 2 = 1

theorem count_four_digit_integers_with_conditions : 
  (Nat.card {n : Nat // is_four_digit_integer n ∧ thousands_digit_is_seven n ∧ hundreds_digit_is_odd n}) = 500 :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_integers_with_conditions_l2205_220578


namespace NUMINAMATH_GPT_john_weekly_earnings_l2205_220538

theorem john_weekly_earnings :
  (4 * 4 * 10 = 160) :=
by
  -- Proposition: John makes $160 a week from streaming
  -- Condition 1: John streams for 4 days a week
  let days_of_streaming := 4
  -- Condition 2: He streams 4 hours each day.
  let hours_per_day := 4
  -- Condition 3: He makes $10 an hour.
  let earnings_per_hour := 10

  -- Now, calculate the weekly earnings
  -- Weekly earnings = 4 days/week * 4 hours/day * $10/hour
  have weekly_earnings : days_of_streaming * hours_per_day * earnings_per_hour = 160 := sorry
  exact weekly_earnings


end NUMINAMATH_GPT_john_weekly_earnings_l2205_220538


namespace NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l2205_220534

-- Let a, b, c, and d be real numbers representing the prices of apples, bananas, cantaloupe, and dates respectively.
variables (a b c d : ℝ)

-- Conditions given in the problem
axiom h1 : a + b + c + d = 40
axiom h2 : d = 3 * a
axiom h3 : c = (a + b) / 2

-- Goal is to prove that the sum of the prices of bananas and cantaloupe is 8 dollars.
theorem cost_of_bananas_and_cantaloupe : b + c = 8 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l2205_220534


namespace NUMINAMATH_GPT_opposite_meaning_for_option_C_l2205_220510

def opposite_meaning (a b : Int) : Bool :=
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem opposite_meaning_for_option_C :
  (opposite_meaning 300 (-500)) ∧ 
  ¬ (opposite_meaning 5 (-5)) ∧ 
  ¬ (opposite_meaning 180 90) ∧ 
  ¬ (opposite_meaning 1 (-1)) :=
by
  unfold opposite_meaning
  sorry

end NUMINAMATH_GPT_opposite_meaning_for_option_C_l2205_220510


namespace NUMINAMATH_GPT_simplify_expression_l2205_220545

variable (q : ℝ)

theorem simplify_expression : ((6 * q + 2) - 3 * q * 5) * 4 + (5 - 2 / 4) * (7 * q - 14) = -4.5 * q - 55 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2205_220545


namespace NUMINAMATH_GPT_volume_of_prism_l2205_220504

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 60)
                                     (h2 : y * z = 75)
                                     (h3 : x * z = 100) :
  x * y * z = 671 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l2205_220504


namespace NUMINAMATH_GPT_lily_lemonade_calories_l2205_220550

def total_weight (lemonade_lime_juice lemonade_honey lemonade_water : ℕ) : ℕ :=
  lemonade_lime_juice + lemonade_honey + lemonade_water

def total_calories (weight_lime_juice weight_honey : ℕ) : ℚ :=
  (30 * weight_lime_juice / 100) + (305 * weight_honey / 100)

def calories_in_portion (total_weight total_calories portion_weight : ℚ) : ℚ :=
  (total_calories * portion_weight) / total_weight

theorem lily_lemonade_calories :
  let lemonade_lime_juice := 150
  let lemonade_honey := 150
  let lemonade_water := 450
  let portion_weight := 300
  let total_weight := total_weight lemonade_lime_juice lemonade_honey lemonade_water
  let total_calories := total_calories lemonade_lime_juice lemonade_honey
  calories_in_portion total_weight total_calories portion_weight = 201 := 
by
  sorry

end NUMINAMATH_GPT_lily_lemonade_calories_l2205_220550


namespace NUMINAMATH_GPT_power_of_x_is_one_l2205_220596

-- The problem setup, defining the existence of distinct primes and conditions on exponents
theorem power_of_x_is_one (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (a b c : ℕ) (h_divisors : (a + 1) * (b + 1) * (c + 1) = 12) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_power_of_x_is_one_l2205_220596


namespace NUMINAMATH_GPT_number_of_ah_tribe_residents_l2205_220522

theorem number_of_ah_tribe_residents 
  (P A U : Nat) 
  (H1 : 16 < P) 
  (H2 : P ≤ 17) 
  (H3 : A + U = P) 
  (H4 : U = 2) : 
  A = 15 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ah_tribe_residents_l2205_220522


namespace NUMINAMATH_GPT_find_a_of_parabola_and_hyperbola_intersection_l2205_220571

theorem find_a_of_parabola_and_hyperbola_intersection
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_asymptotes_intersect_directrix_distance : ∀ (x_A x_B : ℝ),
    -1 / (4 * a) = (1 / 2) * x_A ∧ -1 / (4 * a) = -(1 / 2) * x_B →
    |x_B - x_A| = 4) : a = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_find_a_of_parabola_and_hyperbola_intersection_l2205_220571


namespace NUMINAMATH_GPT_find_specific_M_in_S_l2205_220500

section MatrixProgression

variable {R : Type*} [CommRing R]

-- Definition of arithmetic progression in a 2x2 matrix.
def is_arithmetic_progression (a b c d : R) : Prop :=
  ∃ r : R, b = a + r ∧ c = a + 2 * r ∧ d = a + 3 * r

-- Definition of set S.
def S : Set (Matrix (Fin 2) (Fin 2) R) :=
  { M | ∃ a b c d : R, M = ![![a, b], ![c, d]] ∧ is_arithmetic_progression a b c d }

-- Main problem statement
theorem find_specific_M_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) (k : ℕ) :
  k > 1 → M ∈ S → ∃ (α : ℝ), (M = α • ![![1, 1], ![1, 1]] ∨ (M = α • ![![ -3, -1], ![1, 3]] ∧ Odd k)) :=
by
  sorry

end MatrixProgression

end NUMINAMATH_GPT_find_specific_M_in_S_l2205_220500


namespace NUMINAMATH_GPT_direct_proportion_function_l2205_220515

-- Define the conditions for the problem
def condition1 (m : ℝ) : Prop := m ^ 2 - 1 = 0
def condition2 (m : ℝ) : Prop := m - 1 ≠ 0

-- The main theorem we need to prove
theorem direct_proportion_function (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = -1 :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_function_l2205_220515


namespace NUMINAMATH_GPT_package_cheaper_than_per_person_l2205_220563

theorem package_cheaper_than_per_person (x : ℕ) :
  (90 * 6 + 10 * x < 54 * x + 8 * 3 * x) ↔ x ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_package_cheaper_than_per_person_l2205_220563


namespace NUMINAMATH_GPT_inverse_passes_through_3_4_l2205_220574

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given that f(x) has an inverse
def has_inverse := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Given that y = f(x+1) passes through the point (3,3)
def condition := f (3 + 1) = 3

theorem inverse_passes_through_3_4 
  (h1 : has_inverse f) 
  (h2 : condition f) : 
  f⁻¹ 3 = 4 :=
sorry

end NUMINAMATH_GPT_inverse_passes_through_3_4_l2205_220574


namespace NUMINAMATH_GPT_snail_kite_snails_eaten_l2205_220543

theorem snail_kite_snails_eaten 
  (a₀ : ℕ) (a₁ : ℕ) (a₂ : ℕ) (a₃ : ℕ) (a₄ : ℕ)
  (h₀ : a₀ = 3)
  (h₁ : a₁ = a₀ + 2)
  (h₂ : a₂ = a₁ + 2)
  (h₃ : a₃ = a₂ + 2)
  (h₄ : a₄ = a₃ + 2)
  : a₀ + a₁ + a₂ + a₃ + a₄ = 35 := 
by 
  sorry

end NUMINAMATH_GPT_snail_kite_snails_eaten_l2205_220543


namespace NUMINAMATH_GPT_minimum_value_func_minimum_value_attained_l2205_220539

noncomputable def func (x : ℝ) : ℝ := (4 / (x - 1)) + x

theorem minimum_value_func : ∀ (x : ℝ), x > 1 → func x ≥ 5 :=
by
  intros x hx
  -- proof goes here
  sorry

theorem minimum_value_attained : func 3 = 5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_minimum_value_func_minimum_value_attained_l2205_220539


namespace NUMINAMATH_GPT_value_of_v_l2205_220587

theorem value_of_v (n : ℝ) (v : ℝ) (h1 : 10 * n = v - 2 * n) (h2 : n = -4.5) : v = -9 := by
  sorry

end NUMINAMATH_GPT_value_of_v_l2205_220587


namespace NUMINAMATH_GPT_find_smaller_number_l2205_220581

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l2205_220581


namespace NUMINAMATH_GPT_food_needed_for_vacation_l2205_220514

-- Define the conditions
def daily_food_per_dog := 250 -- in grams
def number_of_dogs := 4
def number_of_days := 14

-- Define the proof problem
theorem food_needed_for_vacation :
  (daily_food_per_dog * number_of_dogs * number_of_days / 1000) = 14 :=
by
  sorry

end NUMINAMATH_GPT_food_needed_for_vacation_l2205_220514


namespace NUMINAMATH_GPT_problem_l2205_220502

theorem problem (a b : ℕ) (h1 : 2^4 + 2^4 = 2^a) (h2 : 3^5 + 3^5 + 3^5 = 3^b) : a + b = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l2205_220502


namespace NUMINAMATH_GPT_number_of_paintings_per_new_gallery_l2205_220589

-- Define all the conditions as variables/constants
def pictures_original : Nat := 9
def new_galleries : Nat := 5
def pencils_per_picture : Nat := 4
def pencils_per_exhibition : Nat := 2
def total_pencils : Nat := 88

-- Define the proof problem in Lean
theorem number_of_paintings_per_new_gallery (pictures_original new_galleries pencils_per_picture pencils_per_exhibition total_pencils : Nat) :
(pictures_original = 9) → (new_galleries = 5) → (pencils_per_picture = 4) → (pencils_per_exhibition = 2) → (total_pencils = 88) → 
∃ (pictures_per_gallery : Nat), pictures_per_gallery = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_paintings_per_new_gallery_l2205_220589


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l2205_220572

-- Define the total number of wheels, wheels on the front axle, 
-- and wheels on each of the other axles.
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4

-- Define the formula for calculating the toll.
def toll_formula (x : ℕ) : ℝ := 2.50 + 0.50 * (x - 2)

-- Calculate the number of other axles.
def calc_other_axles (wheels_left : ℕ) (wheels_per_axle : ℕ) : ℕ :=
wheels_left / wheels_per_axle

-- Statement to prove the final toll is $4.00.
theorem toll_for_18_wheel_truck : toll_formula (
  1 + calc_other_axles (total_wheels - front_axle_wheels) other_axle_wheels
) = 4.00 :=
by sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l2205_220572


namespace NUMINAMATH_GPT_second_person_fraction_removed_l2205_220546

theorem second_person_fraction_removed (teeth_total : ℕ) 
    (removed1 removed3 removed4 : ℕ)
    (total_removed: ℕ)
    (h1: teeth_total = 32)
    (h2: removed1 = teeth_total / 4)
    (h3: removed3 = teeth_total / 2)
    (h4: removed4 = 4)
    (h5 : total_removed = 40):
    ((total_removed - (removed1 + removed3 + removed4)) : ℚ) / teeth_total = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_second_person_fraction_removed_l2205_220546


namespace NUMINAMATH_GPT_rickey_time_l2205_220528

/-- Prejean's speed in a race was three-quarters that of Rickey. 
If they both took a total of 70 minutes to run the race, 
the total number of minutes that Rickey took to finish the race is 40. -/
theorem rickey_time (t : ℝ) (h1 : ∀ p : ℝ, p = (3/4) * t) (h2 : t + (3/4) * t = 70) : t = 40 := 
by
  sorry

end NUMINAMATH_GPT_rickey_time_l2205_220528


namespace NUMINAMATH_GPT_negation_of_existence_lt_zero_l2205_220559

theorem negation_of_existence_lt_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_existence_lt_zero_l2205_220559


namespace NUMINAMATH_GPT_point_coordinates_correct_l2205_220541

def point_coordinates : (ℕ × ℕ) :=
(11, 9)

theorem point_coordinates_correct :
  point_coordinates = (11, 9) :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_correct_l2205_220541


namespace NUMINAMATH_GPT_determine_a_l2205_220542

theorem determine_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x, y) = (-1, 2) → 3 * x + y + a = 0) → ∃ (a : ℝ), a = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l2205_220542


namespace NUMINAMATH_GPT_ratio_avg_speeds_l2205_220547

-- Definitions based on the problem conditions
def distance_A_B := 600
def time_Eddy := 3
def distance_A_C := 460
def time_Freddy := 4

-- Definition of average speeds
def avg_speed_Eddy := distance_A_B / time_Eddy
def avg_speed_Freddy := distance_A_C / time_Freddy

-- Theorem statement
theorem ratio_avg_speeds : avg_speed_Eddy / avg_speed_Freddy = 40 / 23 := 
sorry

end NUMINAMATH_GPT_ratio_avg_speeds_l2205_220547


namespace NUMINAMATH_GPT_geometric_sequence_formula_l2205_220518

theorem geometric_sequence_formula (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 3 / 2)
  (h2 : a 1 + a 1 * q + a 1 * q^2 = 9 / 2)
  (geo : ∀ n, a (n + 1) = a n * q) :
  ∀ n, a n = 3 / 2 * (-2)^(n-1) ∨ a n = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_formula_l2205_220518


namespace NUMINAMATH_GPT_total_cakes_served_l2205_220516

-- Define the conditions
def cakes_lunch_today := 5
def cakes_dinner_today := 6
def cakes_yesterday := 3

-- Define the theorem we want to prove
theorem total_cakes_served : (cakes_lunch_today + cakes_dinner_today + cakes_yesterday) = 14 :=
by
  -- The proof is not required, so we use sorry to skip it
  sorry

end NUMINAMATH_GPT_total_cakes_served_l2205_220516


namespace NUMINAMATH_GPT_combustion_CH₄_forming_water_l2205_220506

/-
Combustion reaction for Methane: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Given:
  3 moles of Methane
  6 moles of Oxygen
  Balanced equation: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Goal: Prove that 6 moles of Water (H₂O) are formed.
-/

-- Define the necessary definitions for the context
def moles_CH₄ : ℝ := 3
def moles_O₂ : ℝ := 6
def ratio_water_methane : ℝ := 2

theorem combustion_CH₄_forming_water :
  moles_CH₄ * ratio_water_methane = 6 :=
by
  sorry

end NUMINAMATH_GPT_combustion_CH₄_forming_water_l2205_220506


namespace NUMINAMATH_GPT_washer_cost_l2205_220585

theorem washer_cost (D : ℝ) (H1 : D + (D + 220) = 1200) : D + 220 = 710 :=
by
  sorry

end NUMINAMATH_GPT_washer_cost_l2205_220585


namespace NUMINAMATH_GPT_quadratic_roots_sign_l2205_220573

theorem quadratic_roots_sign (p q : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x * y = q ∧ x + y = -p) ↔ q < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_sign_l2205_220573


namespace NUMINAMATH_GPT_eggs_from_Martha_is_2_l2205_220533

def eggs_from_Gertrude : ℕ := 4
def eggs_from_Blanche : ℕ := 3
def eggs_from_Nancy : ℕ := 2
def total_eggs_left : ℕ := 9
def eggs_dropped : ℕ := 2

def total_eggs_before_dropping (eggs_from_Martha : ℕ) :=
  eggs_from_Gertrude + eggs_from_Blanche + eggs_from_Nancy + eggs_from_Martha - eggs_dropped = total_eggs_left

-- The theorem stating the eggs collected from Martha.
theorem eggs_from_Martha_is_2 : ∃ (m : ℕ), total_eggs_before_dropping m ∧ m = 2 :=
by
  use 2
  sorry

end NUMINAMATH_GPT_eggs_from_Martha_is_2_l2205_220533


namespace NUMINAMATH_GPT_dina_has_60_dolls_l2205_220570

variable (ivy_collectors_edition_dolls : ℕ)
variable (ivy_total_dolls : ℕ)
variable (dina_dolls : ℕ)

-- Conditions
def condition1 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := ivy_collectors_edition_dolls = 20
def condition2 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := (2 / 3 : ℚ) * ivy_total_dolls = ivy_collectors_edition_dolls
def condition3 (ivy_total_dolls dina_dolls : ℕ) := dina_dolls = 2 * ivy_total_dolls

-- Proof statement
theorem dina_has_60_dolls 
  (h1 : condition1 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h2 : condition2 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h3 : condition3 ivy_total_dolls dina_dolls) : 
  dina_dolls = 60 :=
sorry

end NUMINAMATH_GPT_dina_has_60_dolls_l2205_220570


namespace NUMINAMATH_GPT_meaningful_expression_range_l2205_220568

theorem meaningful_expression_range (x : ℝ) : (∃ y, y = 1 / (x - 4)) ↔ x ≠ 4 := 
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l2205_220568


namespace NUMINAMATH_GPT_probability_of_rolling_five_l2205_220531

-- Define a cube with the given face numbers
def cube_faces : List ℕ := [1, 1, 2, 4, 5, 5]

-- Prove the probability of rolling a "5" is 1/3
theorem probability_of_rolling_five :
  (cube_faces.count 5 : ℚ) / cube_faces.length = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_of_rolling_five_l2205_220531


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2205_220591

-- Definitions from the conditions
def p (a b : ℤ) : Prop := True  -- Since their integrality is given
def q (a b : ℤ) : Prop := ∃ (x : ℤ), (x^2 + a * x + b = 0)

theorem necessary_but_not_sufficient (a b : ℤ) : 
  (¬ (p a b → q a b)) ∧ (q a b → p a b) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2205_220591


namespace NUMINAMATH_GPT_cost_per_set_l2205_220530

variable (C : ℝ)

theorem cost_per_set :
  let total_manufacturing_cost := 10000 + 500 * C
  let revenue := 500 * 50
  let profit := revenue - total_manufacturing_cost
  profit = 5000 → C = 20 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_set_l2205_220530


namespace NUMINAMATH_GPT_prime_base_representation_of_360_l2205_220525

theorem prime_base_representation_of_360 :
  ∃ (exponents : List ℕ), exponents = [3, 2, 1, 0]
  ∧ (2^exponents.head! * 3^(exponents.tail!.head!) * 5^(exponents.tail!.tail!.head!) * 7^(exponents.tail!.tail!.tail!.head!)) = 360 := by
sorry

end NUMINAMATH_GPT_prime_base_representation_of_360_l2205_220525


namespace NUMINAMATH_GPT_prime_factorization_of_expression_l2205_220520

theorem prime_factorization_of_expression :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
sorry

end NUMINAMATH_GPT_prime_factorization_of_expression_l2205_220520


namespace NUMINAMATH_GPT_child_current_height_l2205_220599

variable (h_last_visit : ℝ) (h_grown : ℝ)

-- Conditions
def last_height (h_last_visit : ℝ) := h_last_visit = 38.5
def height_grown (h_grown : ℝ) := h_grown = 3

-- Theorem statement
theorem child_current_height (h_last_visit h_grown : ℝ) 
    (h_last : last_height h_last_visit) 
    (h_grow : height_grown h_grown) : 
    h_last_visit + h_grown = 41.5 :=
by
  sorry

end NUMINAMATH_GPT_child_current_height_l2205_220599


namespace NUMINAMATH_GPT_continuous_at_3_l2205_220535

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 3 then x^2 + x + 2 else 2 * x + a

theorem continuous_at_3 {a : ℝ} : (∀ x : ℝ, 0 < abs (x - 3) → abs (f x a - f 3 a) < 0.0001) →
a = 8 :=
by
  sorry

end NUMINAMATH_GPT_continuous_at_3_l2205_220535


namespace NUMINAMATH_GPT_base_number_mod_100_l2205_220558

theorem base_number_mod_100 (base : ℕ) (h : base ^ 8 % 100 = 1) : base = 1 := 
sorry

end NUMINAMATH_GPT_base_number_mod_100_l2205_220558


namespace NUMINAMATH_GPT_wilson_sledding_l2205_220521

variable (T : ℕ)

theorem wilson_sledding :
  (4 * T) + 6 = 14 → T = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_wilson_sledding_l2205_220521


namespace NUMINAMATH_GPT_cost_price_of_one_ball_is_48_l2205_220554

-- Define the cost price of one ball
def costPricePerBall (x : ℝ) : Prop :=
  let totalCostPrice20Balls := 20 * x
  let sellingPrice20Balls := 720
  let loss := 5 * x
  totalCostPrice20Balls = sellingPrice20Balls + loss

-- Define the main proof problem
theorem cost_price_of_one_ball_is_48 (x : ℝ) (h : costPricePerBall x) : x = 48 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_one_ball_is_48_l2205_220554


namespace NUMINAMATH_GPT_number_of_solutions_l2205_220536

theorem number_of_solutions : 
  ∃ n : ℕ, (∀ x y : ℕ, 3 * x + 4 * y = 766 → x % 2 = 0 → x > 0 → y > 0 → x = n * 2) ∧ n = 127 := 
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l2205_220536


namespace NUMINAMATH_GPT_michelle_oranges_l2205_220549

theorem michelle_oranges (x : ℕ) 
  (h1 : x - x / 3 - 5 = 7) : x = 18 :=
by
  -- We would normally provide the proof here, but it's omitted according to the instructions.
  sorry

end NUMINAMATH_GPT_michelle_oranges_l2205_220549


namespace NUMINAMATH_GPT_negation_of_proposition_l2205_220532

-- Definitions from the problem conditions
def proposition (x : ℝ) := ∃ x < 1, x^2 ≤ 1

-- Reformulated proof problem
theorem negation_of_proposition : 
  ¬ (∃ x < 1, x^2 ≤ 1) ↔ ∀ x < 1, x^2 > 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l2205_220532


namespace NUMINAMATH_GPT_third_discount_l2205_220575

noncomputable def find_discount (P S firstDiscount secondDiscount D3 : ℝ) : Prop :=
  S = P * (1 - firstDiscount / 100) * (1 - secondDiscount / 100) * (1 - D3 / 100)

theorem third_discount (P : ℝ) (S : ℝ) (firstDiscount : ℝ) (secondDiscount : ℝ) (D3 : ℝ) 
  (HP : P = 9649.12) (HS : S = 6600)
  (HfirstDiscount : firstDiscount = 20) (HsecondDiscount : secondDiscount = 10) : 
  find_discount P S firstDiscount secondDiscount 5.01 :=
  by
  rw [HP, HS, HfirstDiscount, HsecondDiscount]
  sorry

end NUMINAMATH_GPT_third_discount_l2205_220575


namespace NUMINAMATH_GPT_transformation_correct_l2205_220584

theorem transformation_correct (a x y : ℝ) (h : a * x = a * y) : 3 - a * x = 3 - a * y :=
sorry

end NUMINAMATH_GPT_transformation_correct_l2205_220584


namespace NUMINAMATH_GPT_problem_statement_l2205_220507

-- Define line and plane as types
variable (Line Plane : Type)

-- Define the perpendicularity and parallelism relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLPlane : Line → Plane → Prop)
variable (perpendicularPPlane : Plane → Plane → Prop)

-- Distinctness of lines and planes
variable (a b : Line)
variable (α β : Plane)

-- Conditions given in the problem
axiom distinct_lines : a ≠ b
axiom distinct_planes : α ≠ β

-- Statement to be proven
theorem problem_statement :
  perpendicular a b → 
  perpendicularLPlane a α → 
  perpendicularLPlane b β → 
  perpendicularPPlane α β :=
sorry

end NUMINAMATH_GPT_problem_statement_l2205_220507


namespace NUMINAMATH_GPT_percent_of_flowers_are_daisies_l2205_220555

-- Definitions for the problem
def total_flowers (F : ℕ) := F
def blue_flowers (F : ℕ) := (7/10) * F
def red_flowers (F : ℕ) := (3/10) * F
def blue_tulips (F : ℕ) := (1/2) * (7/10) * F
def blue_daisies (F : ℕ) := (7/10) * F - (1/2) * (7/10) * F
def red_daisies (F : ℕ) := (2/3) * (3/10) * F
def total_daisies (F : ℕ) := blue_daisies F + red_daisies F
def percentage_of_daisies (F : ℕ) := (total_daisies F / F) * 100

-- The statement to prove
theorem percent_of_flowers_are_daisies (F : ℕ) (hF : F > 0) :
  percentage_of_daisies F = 55 := by
  sorry

end NUMINAMATH_GPT_percent_of_flowers_are_daisies_l2205_220555


namespace NUMINAMATH_GPT_problem_lean_l2205_220577

noncomputable def a : ℕ+ → ℝ := sorry

theorem problem_lean :
  a 11 = 1 / 52 ∧ (∀ n : ℕ+, 1 / a (n + 1) - 1 / a n = 5) → a 1 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_lean_l2205_220577


namespace NUMINAMATH_GPT_smaller_successive_number_l2205_220501

noncomputable def solve_successive_numbers : ℕ :=
  let n := 51
  n

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 2652) : n = solve_successive_numbers :=
  sorry

end NUMINAMATH_GPT_smaller_successive_number_l2205_220501


namespace NUMINAMATH_GPT_net_increase_correct_l2205_220556

-- Definitions for the given conditions
def S1 : ℕ := 10
def B1 : ℕ := 15
def S2 : ℕ := 12
def B2 : ℕ := 8
def S3 : ℕ := 9
def B3 : ℕ := 11

def P1 : ℕ := 250
def P2 : ℕ := 275
def P3 : ℕ := 260
def C1 : ℕ := 100
def C2 : ℕ := 110
def C3 : ℕ := 120

def Sale_profit1 : ℕ := S1 * P1
def Sale_profit2 : ℕ := S2 * P2
def Sale_profit3 : ℕ := S3 * P3

def Repair_cost1 : ℕ := B1 * C1
def Repair_cost2 : ℕ := B2 * C2
def Repair_cost3 : ℕ := B3 * C3

def Net_profit1 : ℕ := Sale_profit1 - Repair_cost1
def Net_profit2 : ℕ := Sale_profit2 - Repair_cost2
def Net_profit3 : ℕ := Sale_profit3 - Repair_cost3

def Total_net_profit : ℕ := Net_profit1 + Net_profit2 + Net_profit3

def Net_Increase : ℕ := (B1 - S1) + (B2 - S2) + (B3 - S3)

-- The theorem to be proven
theorem net_increase_correct : Net_Increase = 3 := by
  sorry

end NUMINAMATH_GPT_net_increase_correct_l2205_220556


namespace NUMINAMATH_GPT_checkerboard_probability_l2205_220594

def checkerboard_size : ℕ := 10

def total_squares (n : ℕ) : ℕ := n * n

def perimeter_squares (n : ℕ) : ℕ := 4 * n - 4

def inner_squares (n : ℕ) : ℕ := total_squares n - perimeter_squares n

def probability_not_touching_edge (n : ℕ) : ℚ := inner_squares n / total_squares n

theorem checkerboard_probability :
  probability_not_touching_edge checkerboard_size = 16 / 25 := by
  sorry

end NUMINAMATH_GPT_checkerboard_probability_l2205_220594


namespace NUMINAMATH_GPT_area_of_f2_equals_7_l2205_220583

def f0 (x : ℝ) : ℝ := abs x
def f1 (x : ℝ) : ℝ := abs (f0 x - 1)
def f2 (x : ℝ) : ℝ := abs (f1 x - 2)

theorem area_of_f2_equals_7 : 
  (∫ x in (-3 : ℝ)..3, f2 x) = 7 :=
by
  sorry

end NUMINAMATH_GPT_area_of_f2_equals_7_l2205_220583


namespace NUMINAMATH_GPT_jill_food_spending_l2205_220597

theorem jill_food_spending :
  ∀ (T : ℝ) (c f o : ℝ),
    c = 0.5 * T →
    o = 0.3 * T →
    (0.04 * c + 0 + 0.1 * o) = 0.05 * T →
    f = 0.2 * T :=
by
  intros T c f o h_c h_o h_tax
  sorry

end NUMINAMATH_GPT_jill_food_spending_l2205_220597


namespace NUMINAMATH_GPT_junk_mail_per_red_or_white_house_l2205_220519

noncomputable def pieces_per_house (total_pieces : ℕ) (total_houses : ℕ) : ℕ := 
  total_pieces / total_houses

noncomputable def total_pieces_for_type (pieces_per_house : ℕ) (houses_of_type : ℕ) : ℕ := 
  pieces_per_house * houses_of_type

noncomputable def total_pieces_for_red_or_white 
  (total_pieces : ℕ)
  (total_houses : ℕ)
  (white_houses : ℕ)
  (red_houses : ℕ) : ℕ :=
  let pieces_per_house := pieces_per_house total_pieces total_houses
  let pieces_for_white := total_pieces_for_type pieces_per_house white_houses
  let pieces_for_red := total_pieces_for_type pieces_per_house red_houses
  pieces_for_white + pieces_for_red

theorem junk_mail_per_red_or_white_house :
  ∀ (total_pieces : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ),
    total_pieces = 48 →
    total_houses = 8 →
    white_houses = 2 →
    red_houses = 3 →
    total_pieces_for_red_or_white total_pieces total_houses white_houses red_houses / (white_houses + red_houses) = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_junk_mail_per_red_or_white_house_l2205_220519


namespace NUMINAMATH_GPT_f_increasing_intervals_g_range_l2205_220576

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1 + Real.sin x) * f x

theorem f_increasing_intervals : 
  (∀ x, 0 ≤ x → x ≤ Real.pi / 2 → 0 ≤ Real.cos x) ∧ (∀ x, 3 * Real.pi / 2 ≤ x → x ≤ 2 * Real.pi → 0 ≤ Real.cos x) :=
sorry

theorem g_range : 
  ∀ x, 0 ≤ x → x ≤ 2 * Real.pi → -1 / 2 ≤ g x ∧ g x ≤ 4 :=
sorry

end NUMINAMATH_GPT_f_increasing_intervals_g_range_l2205_220576


namespace NUMINAMATH_GPT_kim_time_away_from_home_l2205_220588

noncomputable def time_away_from_home (distance_to_friend : ℕ) (detour_percent : ℕ) (stay_time : ℕ) (speed_mph : ℕ) : ℕ :=
  let return_distance := distance_to_friend * (1 + detour_percent / 100)
  let total_distance := distance_to_friend + return_distance
  let driving_time := total_distance / speed_mph
  let driving_time_minutes := driving_time * 60
  driving_time_minutes + stay_time

theorem kim_time_away_from_home : 
  time_away_from_home 30 20 30 44 = 120 := 
by
  -- We will handle the proof here
  sorry

end NUMINAMATH_GPT_kim_time_away_from_home_l2205_220588


namespace NUMINAMATH_GPT_find_n_l2205_220537

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (1 + n)

theorem find_n (k : ℕ) (h : k = 3) (hn : ∃ k, n = k^2)
  (hs : sum_first_n_even_numbers n = 90) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2205_220537


namespace NUMINAMATH_GPT_relationship_between_x_y_l2205_220579

def in_interval (x : ℝ) : Prop := (Real.pi / 4) < x ∧ x < (Real.pi / 2)

noncomputable def x_def (α : ℝ) : ℝ := Real.sin α ^ (Real.log (Real.cos α) / Real.log α)

noncomputable def y_def (α : ℝ) : ℝ := Real.cos α ^ (Real.log (Real.sin α) / Real.log α)

theorem relationship_between_x_y (α : ℝ) (h : in_interval α) : 
  x_def α = y_def α := 
  sorry

end NUMINAMATH_GPT_relationship_between_x_y_l2205_220579


namespace NUMINAMATH_GPT_distinct_roots_polynomial_l2205_220513

theorem distinct_roots_polynomial (a b : ℂ) (h₁ : a ≠ b) (h₂: a^3 + 3*a^2 + a + 1 = 0) (h₃: b^3 + 3*b^2 + b + 1 = 0) :
  a^2 * b + a * b^2 + 3 * a * b = 1 :=
sorry

end NUMINAMATH_GPT_distinct_roots_polynomial_l2205_220513


namespace NUMINAMATH_GPT_rectangle_area_l2205_220565

variables (y w : ℝ)

-- Definitions from conditions
def is_width_of_rectangle : Prop := w = y / Real.sqrt 10
def is_length_of_rectangle : Prop := 3 * w = y / Real.sqrt 10

-- Theorem to be proved
theorem rectangle_area (h1 : is_width_of_rectangle y w) (h2 : is_length_of_rectangle y w) : 
  3 * (w^2) = 3 * (y^2 / 10) :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l2205_220565


namespace NUMINAMATH_GPT_evaluate_expression_l2205_220595

theorem evaluate_expression : 2 + (3 / (4 + (5 / 6))) = 76 / 29 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2205_220595


namespace NUMINAMATH_GPT_change_in_total_berries_l2205_220580

-- Define the initial conditions
def blue_box_berries : ℕ := 35
def increase_diff : ℕ := 100

-- Define the number of strawberries in red boxes
def red_box_berries : ℕ := 100

-- Formulate the change in total number of berries
theorem change_in_total_berries :
  (red_box_berries - blue_box_berries) = 65 :=
by
  have h1 : red_box_berries = increase_diff := rfl
  have h2 : blue_box_berries = 35 := rfl
  rw [h1, h2]
  exact rfl

end NUMINAMATH_GPT_change_in_total_berries_l2205_220580


namespace NUMINAMATH_GPT_coordinates_with_respect_to_origin_l2205_220526

theorem coordinates_with_respect_to_origin (x y : ℤ) (hx : x = 3) (hy : y = -2) : (x, y) = (3, -2) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_with_respect_to_origin_l2205_220526


namespace NUMINAMATH_GPT_solution_is_63_l2205_220598

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)
def last_digit (n : ℕ) : ℕ := n % 10
def rhyming_primes_around (r : ℕ) : Prop :=
  r >= 1 ∧ r <= 100 ∧
  ¬ is_prime r ∧
  ∃ ps : List ℕ, (∀ p ∈ ps, is_prime p ∧ last_digit p = last_digit r) ∧
  (∀ q : ℕ, is_prime q ∧ last_digit q = last_digit r → q ∈ ps) ∧
  List.length ps = 4

theorem solution_is_63 : ∃ r : ℕ, rhyming_primes_around r ∧ r = 63 :=
by sorry

end NUMINAMATH_GPT_solution_is_63_l2205_220598


namespace NUMINAMATH_GPT_evaluate_expression_l2205_220511

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2205_220511


namespace NUMINAMATH_GPT_scientific_notation_of_9280000000_l2205_220561

theorem scientific_notation_of_9280000000 :
  9280000000 = 9.28 * 10^9 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_9280000000_l2205_220561


namespace NUMINAMATH_GPT_product_of_roots_quadratic_l2205_220505

noncomputable def product_of_roots (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  x1 * x2

theorem product_of_roots_quadratic :
  (product_of_roots 1 3 (-5)) = -5 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_quadratic_l2205_220505


namespace NUMINAMATH_GPT_complex_equation_solution_l2205_220569

theorem complex_equation_solution (x y : ℝ)
  (h : (x / (1 - (-ⅈ)) + y / (1 - 2 * (-ⅈ)) = 5 / (1 - 3 * (-ⅈ)))) :
  x + y = 4 :=
sorry

end NUMINAMATH_GPT_complex_equation_solution_l2205_220569


namespace NUMINAMATH_GPT_sequence_8123_appears_l2205_220586

theorem sequence_8123_appears :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) + a (n-4)) % 10) ∧
  (a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ a 4 = 4) ∧
  (∃ n, a n = 8 ∧ a (n+1) = 1 ∧ a (n+2) = 2 ∧ a (n+3) = 3) :=
sorry

end NUMINAMATH_GPT_sequence_8123_appears_l2205_220586


namespace NUMINAMATH_GPT_sum_of_prime_factors_eq_22_l2205_220590

-- Conditions: n is defined as 3^6 - 1
def n : ℕ := 3^6 - 1

-- Statement: The sum of the prime factors of n is 22
theorem sum_of_prime_factors_eq_22 : 
  (∀ p : ℕ, p ∣ n → Prime p → p = 2 ∨ p = 7 ∨ p = 13) → 
  (2 + 7 + 13 = 22) :=
by sorry

end NUMINAMATH_GPT_sum_of_prime_factors_eq_22_l2205_220590


namespace NUMINAMATH_GPT_smallest_positive_debt_pigs_goats_l2205_220562

theorem smallest_positive_debt_pigs_goats :
  ∃ p g : ℤ, 350 * p + 240 * g = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_debt_pigs_goats_l2205_220562


namespace NUMINAMATH_GPT_fractions_with_smallest_difference_l2205_220527

theorem fractions_with_smallest_difference 
    (x y : ℤ) 
    (f1 : ℚ := (x : ℚ) / 8) 
    (f2 : ℚ := (y : ℚ) / 13) 
    (h : abs (13 * x - 8 * y) = 1): 
    (f1 ≠ f2) ∧ abs ((x : ℚ) / 8 - (y : ℚ) / 13) = 1 / 104 :=
by
  sorry

end NUMINAMATH_GPT_fractions_with_smallest_difference_l2205_220527


namespace NUMINAMATH_GPT_find_heaviest_or_lightest_l2205_220509

theorem find_heaviest_or_lightest (stones : Fin 10 → ℝ)
  (h_distinct: ∀ i j : Fin 10, i ≠ j → stones i ≠ stones j)
  (h_pairwise_sums_distinct : ∀ i j k l : Fin 10, 
    i ≠ j → k ≠ l → stones i + stones j ≠ stones k + stones l) :
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≥ stones j) ∨ 
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≤ stones j) :=
sorry

end NUMINAMATH_GPT_find_heaviest_or_lightest_l2205_220509


namespace NUMINAMATH_GPT_relationship_between_abc_l2205_220557

theorem relationship_between_abc (a b c k : ℝ) 
  (hA : -3 = - (k^2 + 1) / a)
  (hB : -2 = - (k^2 + 1) / b)
  (hC : 1 = - (k^2 + 1) / c)
  (hk : 0 < k^2 + 1) : c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_abc_l2205_220557


namespace NUMINAMATH_GPT_gcd_24_36_l2205_220552

theorem gcd_24_36 : Int.gcd 24 36 = 12 := by
  sorry

end NUMINAMATH_GPT_gcd_24_36_l2205_220552


namespace NUMINAMATH_GPT_floor_sum_eq_55_l2205_220540

noncomputable def x : ℝ := 9.42

theorem floor_sum_eq_55 : ∀ (x : ℝ), x = 9.42 → (⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋) = 55 := by
  intros
  sorry

end NUMINAMATH_GPT_floor_sum_eq_55_l2205_220540


namespace NUMINAMATH_GPT_fraction_value_l2205_220592

theorem fraction_value :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l2205_220592


namespace NUMINAMATH_GPT_find_k_for_given_prime_l2205_220593

theorem find_k_for_given_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (k : ℕ) 
  (h : ∃ a : ℕ, k^2 - p * k = a^2) : 
  k = (p + 1)^2 / 4 :=
sorry

end NUMINAMATH_GPT_find_k_for_given_prime_l2205_220593


namespace NUMINAMATH_GPT_no_primes_of_form_2pow5m_plus_2powm_plus_1_l2205_220512

theorem no_primes_of_form_2pow5m_plus_2powm_plus_1 {m : ℕ} (hm : m > 0) : ¬ (Prime (2^(5*m) + 2^m + 1)) :=
by
  sorry

end NUMINAMATH_GPT_no_primes_of_form_2pow5m_plus_2powm_plus_1_l2205_220512


namespace NUMINAMATH_GPT_proof_problem_l2205_220529

open Real

-- Define the problem statements as Lean hypotheses
def p : Prop := ∀ a : ℝ, exp a ≥ a + 1
def q : Prop := ∃ α β : ℝ, sin (α + β) = sin α + sin β

theorem proof_problem : p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2205_220529


namespace NUMINAMATH_GPT_min_value_arith_seq_l2205_220560

noncomputable def S_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem min_value_arith_seq : ∀ n : ℕ, n > 0 → 2 * S_n 2 = (n + 1) * 2 → (n = 4 → (2 * S_n n + 13) / n = 33 / 4) :=
by
  intros n hn hS2 hn_eq_4
  sorry

end NUMINAMATH_GPT_min_value_arith_seq_l2205_220560


namespace NUMINAMATH_GPT_Cindy_hourly_rate_l2205_220548

theorem Cindy_hourly_rate
    (num_courses : ℕ)
    (weekly_hours : ℕ) 
    (monthly_earnings : ℕ) 
    (weeks_in_month : ℕ)
    (monthly_hours_per_course : ℕ)
    (hourly_rate : ℕ) :
    num_courses = 4 →
    weekly_hours = 48 →
    monthly_earnings = 1200 →
    weeks_in_month = 4 →
    monthly_hours_per_course = (weekly_hours / num_courses) * weeks_in_month →
    hourly_rate = monthly_earnings / monthly_hours_per_course →
    hourly_rate = 25 := by
  sorry

end NUMINAMATH_GPT_Cindy_hourly_rate_l2205_220548


namespace NUMINAMATH_GPT_prove_identity_l2205_220553

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end NUMINAMATH_GPT_prove_identity_l2205_220553


namespace NUMINAMATH_GPT_zachary_pushups_l2205_220544

theorem zachary_pushups (david_pushups : ℕ) (h1 : david_pushups = 44) (h2 : ∀ z : ℕ, z = david_pushups + 7) : z = 51 :=
by
  sorry

end NUMINAMATH_GPT_zachary_pushups_l2205_220544
