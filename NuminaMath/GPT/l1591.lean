import Mathlib

namespace NUMINAMATH_GPT_isosceles_triangles_with_perimeter_27_count_l1591_159170

theorem isosceles_triangles_with_perimeter_27_count :
  ∃ n, (∀ (a : ℕ), 7 ≤ a ∧ a ≤ 13 → ∃ (b : ℕ), b = 27 - 2*a ∧ b < 2*a) ∧ n = 7 :=
sorry

end NUMINAMATH_GPT_isosceles_triangles_with_perimeter_27_count_l1591_159170


namespace NUMINAMATH_GPT_total_buckets_poured_l1591_159115

-- Define given conditions
def initial_buckets : ℝ := 1
def additional_buckets : ℝ := 8.8

-- Theorem to prove the total number of buckets poured
theorem total_buckets_poured : 
  initial_buckets + additional_buckets = 9.8 :=
by
  sorry

end NUMINAMATH_GPT_total_buckets_poured_l1591_159115


namespace NUMINAMATH_GPT_perfect_square_formula_l1591_159163

theorem perfect_square_formula (x y : ℝ) :
  ¬∃ a b : ℝ, (x^2 + (1/4)*x + (1/4)) = (a + b)^2 ∧
  ¬∃ c d : ℝ, (x^2 + 2*x*y - y^2) = (c + d)^2 ∧
  ¬∃ e f : ℝ, (x^2 + x*y + y^2) = (e + f)^2 ∧
  ∃ g h : ℝ, (4*x^2 + 4*x + 1) = (g + h)^2 :=
sorry

end NUMINAMATH_GPT_perfect_square_formula_l1591_159163


namespace NUMINAMATH_GPT_terminal_side_of_angle_l1591_159151

theorem terminal_side_of_angle (θ : Real) (h_cos : Real.cos θ < 0) (h_tan : Real.tan θ > 0) :
  θ ∈ {φ : Real | π < φ ∧ φ < 3 * π / 2} :=
sorry

end NUMINAMATH_GPT_terminal_side_of_angle_l1591_159151


namespace NUMINAMATH_GPT_number_of_parakeets_per_cage_l1591_159119

def num_cages : ℕ := 9
def parrots_per_cage : ℕ := 2
def total_birds : ℕ := 72

theorem number_of_parakeets_per_cage : (total_birds - (num_cages * parrots_per_cage)) / num_cages = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_parakeets_per_cage_l1591_159119


namespace NUMINAMATH_GPT_malvina_card_value_sum_l1591_159123

noncomputable def possible_values_sum: ℝ :=
  let value1 := 1
  let value2 := (-1 + Real.sqrt 5) / 2
  (value1 + value2) / 2

theorem malvina_card_value_sum
  (hx : ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ 
                 (x = Real.pi / 4 ∨ (Real.sin x = (-1 + Real.sqrt 5) / 2))):
  possible_values_sum = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_malvina_card_value_sum_l1591_159123


namespace NUMINAMATH_GPT_value_of_sine_neg_10pi_over_3_l1591_159121

theorem value_of_sine_neg_10pi_over_3 : Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_sine_neg_10pi_over_3_l1591_159121


namespace NUMINAMATH_GPT_Carrie_can_add_turnips_l1591_159189

-- Define the variables and conditions
def potatoToTurnipRatio (potatoes turnips : ℕ) : ℚ :=
  potatoes / turnips

def pastPotato : ℕ := 5
def pastTurnip : ℕ := 2
def currentPotato : ℕ := 20
def allowedTurnipAddition : ℕ := 8

-- Define the main theorem to prove, given the conditions.
theorem Carrie_can_add_turnips (past_p_ratio : potatoToTurnipRatio pastPotato pastTurnip = 2.5)
                                : potatoToTurnipRatio currentPotato allowedTurnipAddition = 2.5 :=
sorry

end NUMINAMATH_GPT_Carrie_can_add_turnips_l1591_159189


namespace NUMINAMATH_GPT_option_d_is_true_l1591_159137

theorem option_d_is_true (x : ℝ) : (4 * x) / (x^2 + 4) ≤ 1 := 
  sorry

end NUMINAMATH_GPT_option_d_is_true_l1591_159137


namespace NUMINAMATH_GPT_rightmost_four_digits_of_5_pow_2023_l1591_159110

theorem rightmost_four_digits_of_5_pow_2023 :
  (5 ^ 2023) % 10000 = 8125 :=
sorry

end NUMINAMATH_GPT_rightmost_four_digits_of_5_pow_2023_l1591_159110


namespace NUMINAMATH_GPT_circles_intersect_l1591_159155

-- Definition of the first circle
def circleC := { p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 4 }

-- Definition of the second circle
def circleM := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 9 }

-- Prove that the circles intersect
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, p ∈ circleC ∧ p ∈ circleM := 
sorry

end NUMINAMATH_GPT_circles_intersect_l1591_159155


namespace NUMINAMATH_GPT_find_product_stu_l1591_159158

-- Define hypotheses
variables (a x y c : ℕ)
variables (s t u : ℕ)
variable (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2))

-- Statement to prove the equivalent form and stu product
theorem find_product_stu (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2)) :
  ∃ s t u : ℕ, (a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5 ∧ s * t * u = 12 :=
sorry

end NUMINAMATH_GPT_find_product_stu_l1591_159158


namespace NUMINAMATH_GPT_s_is_arithmetic_progression_l1591_159167

variables (s : ℕ → ℕ) (ds1 ds2 : ℕ)

-- Conditions
axiom strictly_increasing : ∀ n, s n < s (n + 1)
axiom s_is_positive : ∀ n, 0 < s n
axiom s_s_is_arithmetic : ∃ d1, ∀ k, s (s k) = s (s 0) + k * d1
axiom s_s_plus1_is_arithmetic : ∃ d2, ∀ k, s (s k + 1) = s (s 0 + 1) + k * d2

-- Statement to prove
theorem s_is_arithmetic_progression : ∃ d, ∀ k, s (k + 1) = s 0 + k * d :=
sorry

end NUMINAMATH_GPT_s_is_arithmetic_progression_l1591_159167


namespace NUMINAMATH_GPT_y_intercept_of_line_l1591_159166

theorem y_intercept_of_line : ∃ y : ℝ, 4 * 0 + 7 * y = 28 ∧ 0 = 0 ∧ y = 4 := by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1591_159166


namespace NUMINAMATH_GPT_range_of_a_l1591_159113

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1591_159113


namespace NUMINAMATH_GPT_azure_valley_skirts_l1591_159191

variables (P S A : ℕ)

theorem azure_valley_skirts (h1 : P = 10) 
                           (h2 : P = S / 4) 
                           (h3 : S = 2 * A / 3) : 
  A = 60 :=
by sorry

end NUMINAMATH_GPT_azure_valley_skirts_l1591_159191


namespace NUMINAMATH_GPT_Nancy_seeds_l1591_159196

def big_garden_seeds : ℕ := 28
def small_gardens : ℕ := 6
def seeds_per_small_garden : ℕ := 4

def total_seeds : ℕ := big_garden_seeds + small_gardens * seeds_per_small_garden

theorem Nancy_seeds : total_seeds = 52 :=
by
  -- Proof here...
  sorry

end NUMINAMATH_GPT_Nancy_seeds_l1591_159196


namespace NUMINAMATH_GPT_fewest_printers_l1591_159179

theorem fewest_printers (cost1 cost2 : ℕ) (h1 : cost1 = 375) (h2 : cost2 = 150) : 
  ∃ (n : ℕ), n = 2 + 5 :=
by
  have lcm_375_150 : Nat.lcm cost1 cost2 = 750 := sorry
  have n1 : 750 / 375 = 2 := sorry
  have n2 : 750 / 150 = 5 := sorry
  exact ⟨7, rfl⟩

end NUMINAMATH_GPT_fewest_printers_l1591_159179


namespace NUMINAMATH_GPT_find_equation_of_l_l1591_159198

open Real

/-- Define the point M(2, 1) -/
def M : ℝ × ℝ := (2, 1)

/-- Define the original line equation x - 2y + 1 = 0 as a function -/
def line1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- Define the line l that passes through M and is perpendicular to line1 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 5 = 0

/-- The theorem to be proven: the line l passing through M and perpendicular to line1 has the equation 2x + y - 5 = 0 -/
theorem find_equation_of_l (x y : ℝ)
  (hM : M = (2, 1))
  (hl_perpendicular : ∀ x y : ℝ, line1 x y → line_l y (-x / 2)) :
  line_l x y ↔ (x, y) = (2, 1) :=
by
  sorry

end NUMINAMATH_GPT_find_equation_of_l_l1591_159198


namespace NUMINAMATH_GPT_range_of_a_l1591_159153

open Set Real

theorem range_of_a :
  let p := ∀ x : ℝ, |4 * x - 3| ≤ 1
  let q := ∀ x : ℝ, x^2 - (2 * a + 1) * x + (a * (a + 1)) ≤ 0
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q)
  → (∀ x : Icc (0 : ℝ) (1 / 2 : ℝ), a = x) :=
by
  intros
  sorry

end NUMINAMATH_GPT_range_of_a_l1591_159153


namespace NUMINAMATH_GPT_total_cost_fencing_l1591_159100

-- Define the conditions
def length : ℝ := 75
def breadth : ℝ := 25
def cost_per_meter : ℝ := 26.50

-- Define the perimeter of the rectangular plot
def perimeter : ℝ := 2 * length + 2 * breadth

-- Define the total cost of fencing
def total_cost : ℝ := perimeter * cost_per_meter

-- The theorem statement
theorem total_cost_fencing : total_cost = 5300 := 
by 
  -- This is the statement we want to prove
  sorry

end NUMINAMATH_GPT_total_cost_fencing_l1591_159100


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l1591_159169

theorem largest_angle_in_triangle (A B C : ℝ) 
  (a b c : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin C = Real.sqrt 2 * Real.sin B)
  : B = 90 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l1591_159169


namespace NUMINAMATH_GPT_correct_propositions_l1591_159122

theorem correct_propositions (a b c d m : ℝ) :
  (ab > 0 → a > b → (1 / a < 1 / b)) ∧
  (a > |b| → a ^ 2 > b ^ 2) ∧
  ¬ (a > b ∧ c < d → a - d > b - c) ∧
  ¬ (a < b ∧ m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end NUMINAMATH_GPT_correct_propositions_l1591_159122


namespace NUMINAMATH_GPT_min_n_A0_An_ge_200_l1591_159168

theorem min_n_A0_An_ge_200 :
  (∃ n : ℕ, (n * (n + 1)) / 3 ≥ 200) ∧
  (∀ m < 24, (m * (m + 1)) / 3 < 200) :=
sorry

end NUMINAMATH_GPT_min_n_A0_An_ge_200_l1591_159168


namespace NUMINAMATH_GPT_expected_number_of_edges_same_color_3x3_l1591_159107

noncomputable def expected_edges_same_color (board_size : ℕ) (blackened_count : ℕ) : ℚ :=
  let total_pairs := 12       -- 6 horizontal pairs + 6 vertical pairs
  let prob_both_white := 1 / 6
  let prob_both_black := 5 / 18
  let prob_same_color := prob_both_white + prob_both_black
  total_pairs * prob_same_color

theorem expected_number_of_edges_same_color_3x3 :
  expected_edges_same_color 3 5 = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_expected_number_of_edges_same_color_3x3_l1591_159107


namespace NUMINAMATH_GPT_num_valid_pairs_l1591_159138

/-- 
Let S(n) denote the sum of the digits of a natural number n.
Define the predicate to check if the pair (m, n) satisfies the given conditions.
-/
def S (n : ℕ) : ℕ := (toString n).foldl (fun acc ch => acc + ch.toNat - '0'.toNat) 0

def valid_pair (m n : ℕ) : Prop :=
  m < 100 ∧ n < 100 ∧ m > n ∧ m + S n = n + 2 * S m

/-- 
Theorem: There are exactly 99 pairs (m, n) that satisfy the given conditions.
-/
theorem num_valid_pairs : ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 99 ∧
  ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
sorry

end NUMINAMATH_GPT_num_valid_pairs_l1591_159138


namespace NUMINAMATH_GPT_camera_value_l1591_159114

variables (V : ℝ)

def rental_fee_per_week (V : ℝ) := 0.1 * V
def total_rental_fee(V : ℝ) := 4 * rental_fee_per_week V
def johns_share_of_fee(V : ℝ) := 0.6 * (0.4 * total_rental_fee V)

theorem camera_value (h : johns_share_of_fee V = 1200): 
  V = 5000 :=
by
  sorry

end NUMINAMATH_GPT_camera_value_l1591_159114


namespace NUMINAMATH_GPT_pie_eating_fraction_l1591_159103

theorem pie_eating_fraction :
  (1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4 + 1 / 3^5 + 1 / 3^6 + 1 / 3^7) = 1093 / 2187 := 
sorry

end NUMINAMATH_GPT_pie_eating_fraction_l1591_159103


namespace NUMINAMATH_GPT_weight_of_replaced_person_l1591_159178

variable (average_weight_increase : ℝ)
variable (num_persons : ℝ)
variable (weight_new_person : ℝ)

theorem weight_of_replaced_person 
    (h1 : average_weight_increase = 2.5) 
    (h2 : num_persons = 10) 
    (h3 : weight_new_person = 90)
    : ∃ weight_replaced : ℝ, weight_replaced = 65 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l1591_159178


namespace NUMINAMATH_GPT_books_new_arrivals_false_implies_statements_l1591_159149

variable (Books : Type) -- representing the set of books in the library
variable (isNewArrival : Books → Prop) -- predicate stating if a book is a new arrival

theorem books_new_arrivals_false_implies_statements (H : ¬ ∀ b : Books, isNewArrival b) :
  (∃ b : Books, ¬ isNewArrival b) ∧ (¬ ∀ b : Books, isNewArrival b) :=
by
  sorry

end NUMINAMATH_GPT_books_new_arrivals_false_implies_statements_l1591_159149


namespace NUMINAMATH_GPT_jerrys_breakfast_calories_l1591_159182

-- Define the constants based on the conditions
def pancakes : ℕ := 6
def calories_per_pancake : ℕ := 120
def strips_of_bacon : ℕ := 2
def calories_per_strip_of_bacon : ℕ := 100
def calories_in_cereal : ℕ := 200

-- Define the total calories for each category
def total_calories_from_pancakes : ℕ := pancakes * calories_per_pancake
def total_calories_from_bacon : ℕ := strips_of_bacon * calories_per_strip_of_bacon
def total_calories_from_cereal : ℕ := calories_in_cereal

-- Define the total calories in the breakfast
def total_breakfast_calories : ℕ := 
  total_calories_from_pancakes + total_calories_from_bacon + total_calories_from_cereal

-- The theorem we need to prove
theorem jerrys_breakfast_calories : total_breakfast_calories = 1120 := by sorry

end NUMINAMATH_GPT_jerrys_breakfast_calories_l1591_159182


namespace NUMINAMATH_GPT_tin_silver_ratio_l1591_159142

/-- Assuming a metal bar made of an alloy of tin and silver weighs 40 kg, 
    and loses 4 kg in weight when submerged in water,
    where 10 kg of tin loses 1.375 kg in water and 5 kg of silver loses 0.375 kg, 
    prove that the ratio of tin to silver in the bar is 2 : 3. -/
theorem tin_silver_ratio :
  ∃ (T S : ℝ), 
    T + S = 40 ∧ 
    0.1375 * T + 0.075 * S = 4 ∧ 
    T / S = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_tin_silver_ratio_l1591_159142


namespace NUMINAMATH_GPT_smallest_angle_of_triangle_l1591_159111

theorem smallest_angle_of_triangle :
  ∀ a b c : ℝ, a = 2 * Real.sqrt 10 → b = 3 * Real.sqrt 5 → c = 5 → 
  ∃ α β γ : ℝ, α + β + γ = π ∧ α = 45 * (π / 180) ∧ (a = c → α < β ∧ α < γ) ∧ (b = c → β < α ∧ β < γ) ∧ (c = a → γ < α ∧ γ < β) → 
  α = 45 * (π / 180) := 
sorry

end NUMINAMATH_GPT_smallest_angle_of_triangle_l1591_159111


namespace NUMINAMATH_GPT_sample_and_size_correct_l1591_159104

structure SchoolSurvey :=
  (students_selected : ℕ)
  (classes_selected : ℕ)

def survey_sample (survey : SchoolSurvey) : String :=
  "the physical condition of " ++ toString survey.students_selected ++ " students"

def survey_sample_size (survey : SchoolSurvey) : ℕ :=
  survey.students_selected

theorem sample_and_size_correct (survey : SchoolSurvey)
  (h_selected : survey.students_selected = 190)
  (h_classes : survey.classes_selected = 19) :
  survey_sample survey = "the physical condition of 190 students" ∧ 
  survey_sample_size survey = 190 :=
by
  sorry

end NUMINAMATH_GPT_sample_and_size_correct_l1591_159104


namespace NUMINAMATH_GPT_angle_sum_l1591_159109

-- Define the angles in the isosceles triangles
def angle_BAC := 40
def angle_EDF := 50

-- Using the property of isosceles triangles to calculate other angles
def angle_ABC := (180 - angle_BAC) / 2
def angle_DEF := (180 - angle_EDF) / 2

-- Since AD is parallel to CE, angles DAC and ACB are equal as are ADE and DEF
def angle_DAC := angle_ABC
def angle_ADE := angle_DEF

-- The theorem to be proven
theorem angle_sum :
  angle_DAC + angle_ADE = 135 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_l1591_159109


namespace NUMINAMATH_GPT_value_when_x_is_neg1_l1591_159171

theorem value_when_x_is_neg1 (p q : ℝ) (h : p + q = 2022) : 
  (p * (-1)^3 + q * (-1) + 1) = -2021 := by
  sorry

end NUMINAMATH_GPT_value_when_x_is_neg1_l1591_159171


namespace NUMINAMATH_GPT_base7_subtraction_l1591_159116

theorem base7_subtraction (a b : ℕ) (ha : a = 4 * 7^3 + 3 * 7^2 + 2 * 7 + 1)
                            (hb : b = 1 * 7^3 + 2 * 7^2 + 3 * 7 + 4) :
                            a - b = 3 * 7^3 + 0 * 7^2 + 5 * 7 + 4 :=
by
  sorry

end NUMINAMATH_GPT_base7_subtraction_l1591_159116


namespace NUMINAMATH_GPT_gcd_values_count_l1591_159120

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end NUMINAMATH_GPT_gcd_values_count_l1591_159120


namespace NUMINAMATH_GPT_rectangular_garden_length_l1591_159195

theorem rectangular_garden_length (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 900) : l = 300 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_garden_length_l1591_159195


namespace NUMINAMATH_GPT_second_group_men_count_l1591_159124

-- Define the conditions given in the problem
def men1 := 8
def days1 := 80
def days2 := 32

-- The question we need to answer
theorem second_group_men_count : 
  ∃ (men2 : ℕ), men1 * days1 = men2 * days2 ∧ men2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_second_group_men_count_l1591_159124


namespace NUMINAMATH_GPT_max_geometric_sequence_terms_l1591_159108

theorem max_geometric_sequence_terms (a r : ℝ) (n : ℕ) (h_r : r > 1) 
    (h_seq : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 100 ≤ a * r^(k-1) ∧ a * r^(k-1) ≤ 1000) :
  n ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_geometric_sequence_terms_l1591_159108


namespace NUMINAMATH_GPT_max_rectangle_area_l1591_159150

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1591_159150


namespace NUMINAMATH_GPT_simon_legos_l1591_159147

theorem simon_legos (B : ℝ) (K : ℝ) (x : ℝ) (simon_has : ℝ) 
  (h1 : simon_has = B * 1.20)
  (h2 : K = 40)
  (h3 : B = K + x)
  (h4 : simon_has = 72) : simon_has = 72 := by
  sorry

end NUMINAMATH_GPT_simon_legos_l1591_159147


namespace NUMINAMATH_GPT_total_surface_area_l1591_159164

theorem total_surface_area (r h : ℝ) (pi : ℝ) (area_base : ℝ) (curved_area_hemisphere : ℝ) (lateral_area_cylinder : ℝ) :
  (pi * r^2 = 144 * pi) ∧ (h = 10) ∧ (curved_area_hemisphere = 2 * pi * r^2) ∧ (lateral_area_cylinder = 2 * pi * r * h) →
  (curved_area_hemisphere + lateral_area_cylinder + area_base = 672 * pi) :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_l1591_159164


namespace NUMINAMATH_GPT_best_fit_slope_is_correct_l1591_159148

open Real

noncomputable def slope_regression_line (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) :=
  (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21

theorem best_fit_slope_is_correct (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4)
  (h_arith : (x4 - x3 = 2 * (x3 - x2)) ∧ (x3 - x2 = 2 * (x2 - x1))) :
  slope_regression_line x1 x2 x3 x4 y1 y2 y3 y4 = (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21 := 
sorry

end NUMINAMATH_GPT_best_fit_slope_is_correct_l1591_159148


namespace NUMINAMATH_GPT_find_a_and_x_l1591_159156

theorem find_a_and_x (a x : ℝ) (ha1 : x = (2 * a - 1)^2) (ha2 : x = (-a + 2)^2) : a = -1 ∧ x = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_a_and_x_l1591_159156


namespace NUMINAMATH_GPT_manny_paula_weight_l1591_159190

   variable (m n o p : ℕ)

   -- Conditions
   variable (h1 : m + n = 320) 
   variable (h2 : n + o = 295) 
   variable (h3 : o + p = 310) 

   theorem manny_paula_weight : m + p = 335 :=
   by
     sorry
   
end NUMINAMATH_GPT_manny_paula_weight_l1591_159190


namespace NUMINAMATH_GPT_missed_interior_angle_l1591_159105

  theorem missed_interior_angle (n : ℕ) (x : ℝ) 
    (h1 : (n - 2) * 180 = 2750 + x) : x = 130 := 
  by sorry
  
end NUMINAMATH_GPT_missed_interior_angle_l1591_159105


namespace NUMINAMATH_GPT_edward_games_start_l1591_159161

theorem edward_games_start (sold_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h_sold : sold_games = 19) (h_boxes : boxes = 2) (h_game_box : games_per_box = 8) : 
  sold_games + boxes * games_per_box = 35 := 
  by 
    sorry

end NUMINAMATH_GPT_edward_games_start_l1591_159161


namespace NUMINAMATH_GPT_original_book_price_l1591_159199

theorem original_book_price (P : ℝ) (h : 0.85 * P * 1.40 = 476) : P = 476 / (0.85 * 1.40) :=
by
  sorry

end NUMINAMATH_GPT_original_book_price_l1591_159199


namespace NUMINAMATH_GPT_problem_l1591_159159

theorem problem (a b : ℝ) : a^6 + b^6 ≥ a^4 * b^2 + a^2 * b^4 := 
by sorry

end NUMINAMATH_GPT_problem_l1591_159159


namespace NUMINAMATH_GPT_restore_original_problem_l1591_159144

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end NUMINAMATH_GPT_restore_original_problem_l1591_159144


namespace NUMINAMATH_GPT_belfried_industries_payroll_l1591_159128

theorem belfried_industries_payroll (P : ℝ) (tax_paid : ℝ) : 
  ((P > 200000) ∧ (tax_paid = 0.002 * (P - 200000)) ∧ (tax_paid = 200)) → P = 300000 :=
by
  sorry

end NUMINAMATH_GPT_belfried_industries_payroll_l1591_159128


namespace NUMINAMATH_GPT_ratio_w_to_y_l1591_159157

theorem ratio_w_to_y (w x y z : ℝ) (h1 : w / x = 4 / 3) (h2 : y / z = 5 / 3) (h3 : z / x = 1 / 5) :
  w / y = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_w_to_y_l1591_159157


namespace NUMINAMATH_GPT_min_value_expression_l1591_159180

open Real

/-- The minimum value of (14 - x) * (8 - x) * (14 + x) * (8 + x) is -4356. -/
theorem min_value_expression (x : ℝ) : ∃ (a : ℝ), a = (14 - x) * (8 - x) * (14 + x) * (8 + x) ∧ a ≥ -4356 :=
by
  use -4356
  sorry

end NUMINAMATH_GPT_min_value_expression_l1591_159180


namespace NUMINAMATH_GPT_cyclists_meet_at_start_l1591_159143

theorem cyclists_meet_at_start (T : ℚ) (h1 : T = 5 * 7 * 9 / gcd (5 * 7) (gcd (7 * 9) (9 * 5))) : T = 157.5 :=
by
  sorry

end NUMINAMATH_GPT_cyclists_meet_at_start_l1591_159143


namespace NUMINAMATH_GPT_total_bill_l1591_159187

variable (B : ℝ)
variable (h1 : 9 * (B / 10 + 3) = B)

theorem total_bill : B = 270 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_total_bill_l1591_159187


namespace NUMINAMATH_GPT_largest_root_eq_l1591_159175

theorem largest_root_eq : ∃ x, (∀ y, (abs (Real.cos (Real.pi * y) + y^3 - 3 * y^2 + 3 * y) = 3 - y^2 - 2 * y^3) → y ≤ x) ∧ x = 1 := sorry

end NUMINAMATH_GPT_largest_root_eq_l1591_159175


namespace NUMINAMATH_GPT_sequence_term_2012_l1591_159146

theorem sequence_term_2012 :
  ∃ (a : ℕ → ℤ), a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2012 = 6 :=
sorry

end NUMINAMATH_GPT_sequence_term_2012_l1591_159146


namespace NUMINAMATH_GPT_total_boys_in_class_l1591_159162

theorem total_boys_in_class (n : ℕ)
  (h1 : 19 + 19 - 1 = n) :
  n = 37 :=
  sorry

end NUMINAMATH_GPT_total_boys_in_class_l1591_159162


namespace NUMINAMATH_GPT_x_squared_y_minus_xy_squared_l1591_159127

theorem x_squared_y_minus_xy_squared (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) : x^2 * y - x * y^2 = -6 := 
by 
  sorry

end NUMINAMATH_GPT_x_squared_y_minus_xy_squared_l1591_159127


namespace NUMINAMATH_GPT_rattlesnakes_count_l1591_159117

theorem rattlesnakes_count (P B R V : ℕ) (h1 : P = 3 * B / 2) (h2 : V = 2 * 420 / 100) (h3 : P + R = 3 * 420 / 4) (h4 : P + B + R + V = 420) : R = 162 :=
by
  sorry

end NUMINAMATH_GPT_rattlesnakes_count_l1591_159117


namespace NUMINAMATH_GPT_simple_interest_rate_l1591_159176

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (SI_eq : SI = 260)
  (P_eq : P = 910) (T_eq : T = 4)
  (H : SI = P * R * T / 100) : 
  R = 26000 / 3640 := 
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1591_159176


namespace NUMINAMATH_GPT_polynomial_remainder_l1591_159183

theorem polynomial_remainder (P : Polynomial ℝ) (a : ℝ) :
  ∃ (Q : Polynomial ℝ) (r : ℝ), P = Q * (Polynomial.X - Polynomial.C a) + Polynomial.C r ∧ r = (P.eval a) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1591_159183


namespace NUMINAMATH_GPT_problem_equivalence_l1591_159160

section ProblemDefinitions

def odd_function_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def statement_A (f : ℝ → ℝ) : Prop :=
  (∀ x < 0, f x = -Real.log (-x)) →
  odd_function_condition f →
  ∀ x > 0, f x ≠ -Real.log x

def statement_B (a : ℝ) : Prop :=
  Real.logb a (1 / 2) < 1 →
  (0 < a ∧ a < 1 / 2) ∨ (1 < a)

def statement_C : Prop :=
  ∀ x, (Real.logb 2 (Real.sqrt (x-1)) = (1/2) * Real.logb 2 x)

def statement_D (x1 x2 : ℝ) : Prop :=
  (x1 + Real.log x1 = 2) →
  (Real.log (1 - x2) - x2 = 1) →
  x1 + x2 = 1

end ProblemDefinitions

structure MathProofProblem :=
  (A : ∀ f : ℝ → ℝ, statement_A f)
  (B : ∀ a : ℝ, statement_B a)
  (C : statement_C)
  (D : ∀ x1 x2 : ℝ, statement_D x1 x2)

theorem problem_equivalence : MathProofProblem :=
  { A := sorry,
    B := sorry,
    C := sorry,
    D := sorry }

end NUMINAMATH_GPT_problem_equivalence_l1591_159160


namespace NUMINAMATH_GPT_equivalence_gcd_prime_power_l1591_159118

theorem equivalence_gcd_prime_power (a b n : ℕ) :
  (∀ m, 0 < m ∧ m < n → Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔ 
  (∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k) :=
by
  sorry

end NUMINAMATH_GPT_equivalence_gcd_prime_power_l1591_159118


namespace NUMINAMATH_GPT_p_computation_l1591_159134

def p (x y : Int) : Int :=
  if x >= 0 ∧ y >= 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x + y > 0 then 2 * x + 2 * y
  else x + 4 * y

theorem p_computation : p (p 2 (-3)) (p (-3) (-4)) = 26 := by
  sorry

end NUMINAMATH_GPT_p_computation_l1591_159134


namespace NUMINAMATH_GPT_sum_of_dimensions_l1591_159126

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 30) (h2 : A * C = 60) (h3 : B * C = 90) : A + B + C = 24 := 
sorry

end NUMINAMATH_GPT_sum_of_dimensions_l1591_159126


namespace NUMINAMATH_GPT_prob_both_hit_prob_at_least_one_hits_l1591_159188

variable (pA pB : ℝ)

-- Given conditions
def prob_A_hits : Prop := pA = 0.9
def prob_B_hits : Prop := pB = 0.8

-- Proof problems
theorem prob_both_hit (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  pA * pB = 0.72 := 
  sorry

theorem prob_at_least_one_hits (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  1 - (1 - pA) * (1 - pB) = 0.98 := 
  sorry

end NUMINAMATH_GPT_prob_both_hit_prob_at_least_one_hits_l1591_159188


namespace NUMINAMATH_GPT_diagonal_of_rectangular_solid_l1591_159193

-- Define the lengths of the edges
def a : ℝ := 2
def b : ℝ := 3
def c : ℝ := 4

-- Prove that the diagonal of the rectangular solid with edges a, b, and c is sqrt(29)
theorem diagonal_of_rectangular_solid (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : (a^2 + b^2 + c^2) = 29 := 
by 
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_diagonal_of_rectangular_solid_l1591_159193


namespace NUMINAMATH_GPT_polygon_diagonals_with_one_non_connecting_vertex_l1591_159125

-- Define the number of sides in the polygon
def num_sides : ℕ := 17

-- Define the formula to calculate the number of diagonals in a polygon
def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the number of non-connecting vertex to any diagonal
def non_connected_diagonals (n : ℕ) : ℕ :=
  n - 3

-- The theorem to state and prove
theorem polygon_diagonals_with_one_non_connecting_vertex :
  total_diagonals num_sides - non_connected_diagonals num_sides = 105 :=
by
  -- The formal proof would go here
  sorry

end NUMINAMATH_GPT_polygon_diagonals_with_one_non_connecting_vertex_l1591_159125


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1591_159133

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ x^2 + a * x + b < 0) : b = 2 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1591_159133


namespace NUMINAMATH_GPT_fraction_under_11_is_one_third_l1591_159181

def fraction_under_11 (T : ℕ) (fraction_above_11_under_13 : ℚ) (students_above_13 : ℕ) : ℚ :=
  let fraction_under_11 := 1 - (fraction_above_11_under_13 + students_above_13 / T)
  fraction_under_11

theorem fraction_under_11_is_one_third :
  fraction_under_11 45 (2/5) 12 = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_under_11_is_one_third_l1591_159181


namespace NUMINAMATH_GPT_vartan_recreation_percent_l1591_159186

noncomputable def percent_recreation_week (last_week_wages current_week_wages current_week_recreation last_week_recreation : ℝ) : ℝ :=
  (current_week_recreation / current_week_wages) * 100

theorem vartan_recreation_percent 
  (W : ℝ) 
  (h1 : last_week_wages = W)  
  (h2 : last_week_recreation = 0.15 * W)
  (h3 : current_week_wages = 0.90 * W)
  (h4 : current_week_recreation = 1.80 * last_week_recreation) :
  percent_recreation_week last_week_wages current_week_wages current_week_recreation last_week_recreation = 30 :=
by
  sorry

end NUMINAMATH_GPT_vartan_recreation_percent_l1591_159186


namespace NUMINAMATH_GPT_reciprocal_of_neg_five_l1591_159106

theorem reciprocal_of_neg_five: 
  ∃ x : ℚ, -5 * x = 1 ∧ x = -1 / 5 := 
sorry

end NUMINAMATH_GPT_reciprocal_of_neg_five_l1591_159106


namespace NUMINAMATH_GPT_ratio_lions_l1591_159140

variable (Safari_Lions : Nat)
variable (Safari_Snakes : Nat)
variable (Safari_Giraffes : Nat)
variable (Savanna_Lions_Ratio : ℕ)
variable (Savanna_Snakes : Nat)
variable (Savanna_Giraffes : Nat)
variable (Savanna_Total : Nat)

-- Conditions
def conditions := 
  (Safari_Lions = 100) ∧
  (Safari_Snakes = Safari_Lions / 2) ∧
  (Safari_Giraffes = Safari_Snakes - 10) ∧
  (Savanna_Lions_Ratio * Safari_Lions + Savanna_Snakes + Savanna_Giraffes = Savanna_Total) ∧
  (Savanna_Snakes = 3 * Safari_Snakes) ∧
  (Savanna_Giraffes = Safari_Giraffes + 20) ∧
  (Savanna_Total = 410)

-- Theorem to prove
theorem ratio_lions : conditions Safari_Lions Safari_Snakes Safari_Giraffes Savanna_Lions_Ratio Savanna_Snakes Savanna_Giraffes Savanna_Total → Savanna_Lions_Ratio = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_lions_l1591_159140


namespace NUMINAMATH_GPT_total_simple_interest_l1591_159102

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest : simple_interest 2500 10 4 = 1000 := 
by
  sorry

end NUMINAMATH_GPT_total_simple_interest_l1591_159102


namespace NUMINAMATH_GPT_german_mo_2016_problem_1_l1591_159131

theorem german_mo_2016_problem_1 (a b : ℝ) :
  a^2 + b^2 = 25 ∧ 3 * (a + b) - a * b = 15 ↔
  (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨
  (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4) :=
sorry

end NUMINAMATH_GPT_german_mo_2016_problem_1_l1591_159131


namespace NUMINAMATH_GPT_certain_event_is_A_l1591_159197

def isCertainEvent (event : Prop) : Prop := event

axiom event_A : Prop
axiom event_B : Prop
axiom event_C : Prop
axiom event_D : Prop

axiom event_A_is_certain : isCertainEvent event_A
axiom event_B_is_not_certain : ¬ isCertainEvent event_B
axiom event_C_is_impossible : ¬ event_C
axiom event_D_is_not_certain : ¬ isCertainEvent event_D

theorem certain_event_is_A : isCertainEvent event_A := by
  exact event_A_is_certain

end NUMINAMATH_GPT_certain_event_is_A_l1591_159197


namespace NUMINAMATH_GPT_dave_deleted_apps_l1591_159194

-- Definitions based on problem conditions
def original_apps : Nat := 16
def remaining_apps : Nat := 5

-- Theorem statement for proving how many apps Dave deleted
theorem dave_deleted_apps : original_apps - remaining_apps = 11 :=
by
  sorry

end NUMINAMATH_GPT_dave_deleted_apps_l1591_159194


namespace NUMINAMATH_GPT_xy_is_perfect_cube_l1591_159129

theorem xy_is_perfect_cube (x y : ℕ) (h₁ : x = 5 * 2^4 * 3^3) (h₂ : y = 2^2 * 5^2) : ∃ z : ℕ, (x * y) = z^3 :=
by
  sorry

end NUMINAMATH_GPT_xy_is_perfect_cube_l1591_159129


namespace NUMINAMATH_GPT_insects_total_l1591_159184

def total_insects (n_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
                  (n_stones : ℕ) (ants_per_stone : ℕ) 
                  (total_bees : ℕ) (n_flowers : ℕ) : ℕ :=
  let num_ladybugs := n_leaves * ladybugs_per_leaf
  let num_ants := n_stones * ants_per_stone
  let num_bees := total_bees -- already given as total_bees
  num_ladybugs + num_ants + num_bees

theorem insects_total : total_insects 345 267 178 423 498 6 = 167967 :=
  by unfold total_insects; sorry

end NUMINAMATH_GPT_insects_total_l1591_159184


namespace NUMINAMATH_GPT_ratio_simplified_l1591_159177

theorem ratio_simplified (kids_meals : ℕ) (adult_meals : ℕ) (h1 : kids_meals = 70) (h2 : adult_meals = 49) : 
  ∃ (k a : ℕ), k = 10 ∧ a = 7 ∧ kids_meals / Nat.gcd kids_meals adult_meals = k ∧ adult_meals / Nat.gcd kids_meals adult_meals = a :=
by
  sorry

end NUMINAMATH_GPT_ratio_simplified_l1591_159177


namespace NUMINAMATH_GPT_bread_rise_time_l1591_159141

theorem bread_rise_time (x : ℕ) (kneading_time : ℕ) (baking_time : ℕ) (total_time : ℕ) 
  (h1 : kneading_time = 10) 
  (h2 : baking_time = 30) 
  (h3 : total_time = 280) 
  (h4 : kneading_time + baking_time + 2 * x = total_time) : 
  x = 120 :=
sorry

end NUMINAMATH_GPT_bread_rise_time_l1591_159141


namespace NUMINAMATH_GPT_find_m_l1591_159174

theorem find_m (x y m : ℤ) 
  (h1 : 4 * x + y = 34)
  (h2 : m * x - y = 20)
  (h3 : y ^ 2 = 4) 
  : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_l1591_159174


namespace NUMINAMATH_GPT_determine_n_l1591_159154

noncomputable def polynomial (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for the actual polynomial function

theorem determine_n (n : ℕ) 
  (h_deg : ∀ a, polynomial n a = 2 → (3 ∣ a) ∨ a = 0)
  (h_deg' : ∀ a, polynomial n a = 1 → (3 ∣ (a + 2)))
  (h_deg'' : ∀ a, polynomial n a = 0 → (3 ∣ (a + 1)))
  (h_val : polynomial n (3*n+1) = 730) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_determine_n_l1591_159154


namespace NUMINAMATH_GPT_Grant_room_count_l1591_159135

-- Defining the number of rooms in each person's apartments
def Danielle_rooms : ℕ := 6
def Heidi_rooms : ℕ := 3 * Danielle_rooms
def Jenny_rooms : ℕ := Danielle_rooms + 5

-- Combined total rooms
def Total_rooms : ℕ := Danielle_rooms + Heidi_rooms + Jenny_rooms

-- Division operation to determine Grant's room count
def Grant_rooms (total_rooms : ℕ) : ℕ := total_rooms / 9

-- Statement to be proved
theorem Grant_room_count : Grant_rooms Total_rooms = 3 := by
  sorry

end NUMINAMATH_GPT_Grant_room_count_l1591_159135


namespace NUMINAMATH_GPT_DeepakAgeProof_l1591_159173

def RahulAgeAfter10Years (RahulAge : ℕ) : Prop := RahulAge + 10 = 26

def DeepakPresentAge (ratioRahul ratioDeepak : ℕ) (RahulAge : ℕ) : ℕ :=
  (2 * RahulAge) / ratioRahul

theorem DeepakAgeProof {DeepakCurrentAge : ℕ}
  (ratioRahul ratioDeepak RahulAge : ℕ)
  (hRatio : ratioRahul = 4)
  (hDeepakRatio : ratioDeepak = 2) :
  RahulAgeAfter10Years RahulAge →
  DeepakCurrentAge = DeepakPresentAge ratioRahul ratioDeepak RahulAge :=
  sorry

end NUMINAMATH_GPT_DeepakAgeProof_l1591_159173


namespace NUMINAMATH_GPT_find_m_l1591_159192

variable {S : ℕ → ℤ}
variable {m : ℕ}

/-- Given the sequences conditions, the value of m is 5 --/
theorem find_m (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) (h4 : 2 ≤ m) : m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_l1591_159192


namespace NUMINAMATH_GPT_train_speed_in_kmh_l1591_159165

-- Definitions from the conditions
def length_of_train : ℝ := 800 -- in meters
def time_to_cross_pole : ℝ := 20 -- in seconds
def conversion_factor : ℝ := 3.6 -- (km/h) per (m/s)

-- Statement to prove the train's speed in km/h
theorem train_speed_in_kmh :
  (length_of_train / time_to_cross_pole * conversion_factor) = 144 :=
  sorry

end NUMINAMATH_GPT_train_speed_in_kmh_l1591_159165


namespace NUMINAMATH_GPT_total_space_needed_for_trees_l1591_159145

def appleTreeWidth : ℕ := 10
def spaceBetweenAppleTrees : ℕ := 12
def numAppleTrees : ℕ := 2

def peachTreeWidth : ℕ := 12
def spaceBetweenPeachTrees : ℕ := 15
def numPeachTrees : ℕ := 2

def totalSpace : ℕ :=
  numAppleTrees * appleTreeWidth + spaceBetweenAppleTrees +
  numPeachTrees * peachTreeWidth + spaceBetweenPeachTrees

theorem total_space_needed_for_trees : totalSpace = 71 := by
  sorry

end NUMINAMATH_GPT_total_space_needed_for_trees_l1591_159145


namespace NUMINAMATH_GPT_problem_statement_l1591_159132

def operation (a b : ℝ) := (a + b) ^ 2

theorem problem_statement (x y : ℝ) : operation ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1591_159132


namespace NUMINAMATH_GPT_smallest_b_for_factoring_l1591_159112

theorem smallest_b_for_factoring (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + (1200 : ℤ) = (x + r)*(x + s) ∧ b = r + s ∧ r * s = 1200) →
  b = 70 := 
sorry

end NUMINAMATH_GPT_smallest_b_for_factoring_l1591_159112


namespace NUMINAMATH_GPT_domain_of_f_lg_x_l1591_159172

theorem domain_of_f_lg_x : 
  ({x : ℝ | -1 ≤ x ∧ x ≤ 1} = {x | 10 ≤ x ∧ x ≤ 100}) ↔ (∃ f : ℝ → ℝ, ∀ x ∈ {x : ℝ | -1 ≤ x ∧ x ≤ 1}, f (x * x + 1) = f (Real.log x)) :=
sorry

end NUMINAMATH_GPT_domain_of_f_lg_x_l1591_159172


namespace NUMINAMATH_GPT_sweets_ratio_l1591_159139

theorem sweets_ratio (x : ℕ) (h1 : x + 4 + 7 = 22) : x / 22 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sweets_ratio_l1591_159139


namespace NUMINAMATH_GPT_sum_of_terms_l1591_159136

def sequence_sum (n : ℕ) : ℕ :=
  n^2 + 2*n + 5

theorem sum_of_terms : sequence_sum 9 - sequence_sum 6 = 51 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_terms_l1591_159136


namespace NUMINAMATH_GPT_drone_height_l1591_159101

theorem drone_height (TR TS TU : ℝ) (UR : TU^2 + TR^2 = 180^2) (US : TU^2 + TS^2 = 150^2) (RS : TR^2 + TS^2 = 160^2) : 
  TU = Real.sqrt 14650 :=
by
  sorry

end NUMINAMATH_GPT_drone_height_l1591_159101


namespace NUMINAMATH_GPT_part1_part2_l1591_159185

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x : ℝ) := x^2 - 1

theorem part1 {x : ℝ} (h : 1 ≤ x) : f x ≤ (1 / 2) * g x := by
  sorry

theorem part2 {m : ℝ} : (∀ x, 1 ≤ x → f x - m * g x ≤ 0) → m ≥ (1 / 2) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1591_159185


namespace NUMINAMATH_GPT_rectangle_vertex_area_y_value_l1591_159152

theorem rectangle_vertex_area_y_value (y : ℕ) (hy : 0 ≤ y) :
  let A := (0, y)
  let B := (10, y)
  let C := (0, 4)
  let D := (10, 4)
  10 * (y - 4) = 90 → y = 13 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_vertex_area_y_value_l1591_159152


namespace NUMINAMATH_GPT_find_x_plus_inv_x_l1591_159130

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_plus_inv_x_l1591_159130
