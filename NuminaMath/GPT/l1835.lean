import Mathlib

namespace NUMINAMATH_GPT_intersection_of_domains_l1835_183551

def M (x : ℝ) : Prop := x < 1
def N (x : ℝ) : Prop := x > -1
def P (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem intersection_of_domains : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | P x} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_domains_l1835_183551


namespace NUMINAMATH_GPT_min_reciprocal_sum_l1835_183561

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_reciprocal_sum_l1835_183561


namespace NUMINAMATH_GPT_cannot_fit_480_pictures_l1835_183516

theorem cannot_fit_480_pictures 
  (A_capacity : ℕ) (B_capacity : ℕ) (C_capacity : ℕ) 
  (n_A : ℕ) (n_B : ℕ) (n_C : ℕ) 
  (total_pictures : ℕ) : 
  A_capacity = 12 → B_capacity = 18 → C_capacity = 24 → 
  n_A = 6 → n_B = 4 → n_C = 3 → 
  total_pictures = 480 → 
  A_capacity * n_A + B_capacity * n_B + C_capacity * n_C < total_pictures :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_cannot_fit_480_pictures_l1835_183516


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l1835_183519

/-- 
  Given 200 girls and a total of 600 students in a college,
  the ratio of the number of boys to the number of girls is 2:1.
--/
theorem ratio_of_boys_to_girls 
  (num_girls : ℕ) (total_students : ℕ) (h_girls : num_girls = 200) 
  (h_total : total_students = 600) : 
  (total_students - num_girls) / num_girls = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l1835_183519


namespace NUMINAMATH_GPT_supplies_total_cost_l1835_183548

-- Definitions based on conditions in a)
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def cost_of_baking_soda : ℕ := 1
def students_count : ℕ := 23

-- The main theorem to prove
theorem supplies_total_cost :
  cost_of_bow * students_count + cost_of_vinegar * students_count + cost_of_baking_soda * students_count = 184 :=
by
  sorry

end NUMINAMATH_GPT_supplies_total_cost_l1835_183548


namespace NUMINAMATH_GPT_cattle_selling_price_per_pound_correct_l1835_183594

def purchase_price : ℝ := 40000
def cattle_count : ℕ := 100
def feed_cost_percentage : ℝ := 0.20
def weight_per_head : ℕ := 1000
def profit : ℝ := 112000

noncomputable def total_feed_cost : ℝ := purchase_price * feed_cost_percentage
noncomputable def total_cost : ℝ := purchase_price + total_feed_cost
noncomputable def total_revenue : ℝ := total_cost + profit
def total_weight : ℕ := cattle_count * weight_per_head
noncomputable def selling_price_per_pound : ℝ := total_revenue / total_weight

theorem cattle_selling_price_per_pound_correct :
  selling_price_per_pound = 1.60 := by
  sorry

end NUMINAMATH_GPT_cattle_selling_price_per_pound_correct_l1835_183594


namespace NUMINAMATH_GPT_find_inscribed_circle_area_l1835_183579

noncomputable def inscribed_circle_area (length : ℝ) (breadth : ℝ) : ℝ :=
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let radius_circle := side_square / 2
  Real.pi * radius_circle^2

theorem find_inscribed_circle_area :
  inscribed_circle_area 36 28 = 804.25 := by
  sorry

end NUMINAMATH_GPT_find_inscribed_circle_area_l1835_183579


namespace NUMINAMATH_GPT_initial_fish_count_l1835_183530

theorem initial_fish_count (x : ℕ) (h1 : x + 47 = 69) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_initial_fish_count_l1835_183530


namespace NUMINAMATH_GPT_base_conversion_l1835_183542

def baseThreeToBaseTen (n : List ℕ) : ℕ :=
  n.reverse.enumFrom 0 |>.map (λ ⟨i, d⟩ => d * 3^i) |>.sum

def baseTenToBaseFive (n : ℕ) : List ℕ :=
  let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else aux (n / 5) ((n % 5) :: acc)
  aux n []

theorem base_conversion (baseThreeNum : List ℕ) (baseTenNum : ℕ) (baseFiveNum : List ℕ) :
  baseThreeNum = [2, 0, 1, 2, 1] →
  baseTenNum = 178 →
  baseFiveNum = [1, 2, 0, 3] →
  baseThreeToBaseTen baseThreeNum = baseTenNum ∧ baseTenToBaseFive baseTenNum = baseFiveNum :=
by
  intros h1 h2 h3
  unfold baseThreeToBaseTen
  unfold baseTenToBaseFive
  sorry

end NUMINAMATH_GPT_base_conversion_l1835_183542


namespace NUMINAMATH_GPT_distribute_marbles_correct_l1835_183544

def distribute_marbles (total_marbles : Nat) (num_boys : Nat) : Nat :=
  total_marbles / num_boys

theorem distribute_marbles_correct :
  distribute_marbles 20 2 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_distribute_marbles_correct_l1835_183544


namespace NUMINAMATH_GPT_student_sums_l1835_183526

theorem student_sums (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 48) : y = 36 :=
by
  sorry

end NUMINAMATH_GPT_student_sums_l1835_183526


namespace NUMINAMATH_GPT_greatest_number_zero_l1835_183554

-- Define the condition (inequality)
def inequality (x : ℤ) : Prop :=
  3 * x + 2 < 5 - 2 * x

-- Define the property of being the greatest whole number satisfying the inequality
def greatest_whole_number (x : ℤ) : Prop :=
  inequality x ∧ (∀ y : ℤ, inequality y → y ≤ x)

-- The main theorem stating the greatest whole number satisfying the inequality is 0
theorem greatest_number_zero : greatest_whole_number 0 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_zero_l1835_183554


namespace NUMINAMATH_GPT_rectangle_length_reduction_l1835_183596

theorem rectangle_length_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_length := L * (1 - 10 / 100)
  let new_width := W * (10 / 9)
  (new_length * new_width = L * W) → 
  x = 10 := by sorry

end NUMINAMATH_GPT_rectangle_length_reduction_l1835_183596


namespace NUMINAMATH_GPT_farmer_land_l1835_183557

theorem farmer_land (A : ℝ) (A_nonneg : A ≥ 0) (cleared_land : ℝ) 
  (soybeans wheat potatoes vegetables corn : ℝ) 
  (h_cleared : cleared_land = 0.95 * A) 
  (h_soybeans : soybeans = 0.35 * cleared_land) 
  (h_wheat : wheat = 0.40 * cleared_land) 
  (h_potatoes : potatoes = 0.15 * cleared_land) 
  (h_vegetables : vegetables = 0.08 * cleared_land) 
  (h_corn : corn = 630) 
  (cleared_sum : soybeans + wheat + potatoes + vegetables + corn = cleared_land) :
  A = 33158 := 
by 
  sorry

end NUMINAMATH_GPT_farmer_land_l1835_183557


namespace NUMINAMATH_GPT_difference_between_heads_and_feet_l1835_183520

-- Definitions based on the conditions
def penguins := 30
def zebras := 22
def tigers := 8
def zookeepers := 12

-- Counting heads
def heads := penguins + zebras + tigers + zookeepers

-- Counting feet
def feet := (2 * penguins) + (4 * zebras) + (4 * tigers) + (2 * zookeepers)

-- Proving the difference between the number of feet and heads is 132
theorem difference_between_heads_and_feet : (feet - heads) = 132 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_heads_and_feet_l1835_183520


namespace NUMINAMATH_GPT_remainder_p_x_minus_2_l1835_183521

def p (x : ℝ) := x^5 + 2 * x^2 + 3

theorem remainder_p_x_minus_2 : p 2 = 43 := 
by
  sorry

end NUMINAMATH_GPT_remainder_p_x_minus_2_l1835_183521


namespace NUMINAMATH_GPT_valid_number_count_is_300_l1835_183527

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6]

-- Define the set of odd digits
def odd_digits : List ℕ := [1, 3, 5]

-- Define a function to count valid four-digit numbers
noncomputable def count_valid_numbers : ℕ :=
  (odd_digits.length * (digits.length - 2) * (digits.length - 2) * (digits.length - 3))

-- State the theorem
theorem valid_number_count_is_300 : count_valid_numbers = 300 :=
  sorry

end NUMINAMATH_GPT_valid_number_count_is_300_l1835_183527


namespace NUMINAMATH_GPT_speed_with_stream_l1835_183568

-- Definitions for the conditions in part a
def Vm : ℕ := 8  -- Speed of the man in still water (in km/h)
def Vs : ℕ := Vm - 4  -- Speed of the stream (in km/h), derived from man's speed against the stream

-- The statement to prove the man's speed with the stream
theorem speed_with_stream : Vm + Vs = 12 := by sorry

end NUMINAMATH_GPT_speed_with_stream_l1835_183568


namespace NUMINAMATH_GPT_cookies_ratio_l1835_183588

theorem cookies_ratio (total_cookies sells_mr_stone brock_buys left_cookies katy_buys : ℕ)
  (h1 : total_cookies = 5 * 12)
  (h2 : sells_mr_stone = 2 * 12)
  (h3 : brock_buys = 7)
  (h4 : left_cookies = 15)
  (h5 : total_cookies - sells_mr_stone - brock_buys - left_cookies = katy_buys) :
  katy_buys / brock_buys = 2 :=
by sorry

end NUMINAMATH_GPT_cookies_ratio_l1835_183588


namespace NUMINAMATH_GPT_selling_price_of_book_l1835_183500

theorem selling_price_of_book (SP : ℝ) (CP : ℝ := 200) :
  (SP - CP) = (340 - CP) + 0.05 * CP → SP = 350 :=
by {
  sorry
}

end NUMINAMATH_GPT_selling_price_of_book_l1835_183500


namespace NUMINAMATH_GPT_polar_r_eq_3_is_circle_l1835_183524

theorem polar_r_eq_3_is_circle :
  ∀ θ : ℝ, ∃ x y : ℝ, (x, y) = (3 * Real.cos θ, 3 * Real.sin θ) ∧ x^2 + y^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_polar_r_eq_3_is_circle_l1835_183524


namespace NUMINAMATH_GPT_find_a₃_l1835_183552

variable (a₁ a₂ a₃ a₄ a₅ : ℝ)
variable (S₅ : ℝ) (a_seq : ℕ → ℝ)

-- Define the conditions for arithmetic sequence and given sum
def is_arithmetic_sequence (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_seq (n+1) - a_seq n = a_seq 1 - a_seq 0

axiom sum_first_five_terms (S₅ : ℝ) (hS : S₅ = 20) : 
  S₅ = (5 * (a₁ + a₅)) / 2

-- Main theorem we need to prove
theorem find_a₃ (hS₅ : S₅ = 20) (h_seq : is_arithmetic_sequence a_seq) :
  (∃ (a₃ : ℝ), a₃ = 4) :=
sorry

end NUMINAMATH_GPT_find_a₃_l1835_183552


namespace NUMINAMATH_GPT_two_digit_integer_eq_55_l1835_183536

theorem two_digit_integer_eq_55
  (c : ℕ)
  (h1 : c / 10 + c % 10 = 10)
  (h2 : (c / 10) * (c % 10) = 25) :
  c = 55 :=
  sorry

end NUMINAMATH_GPT_two_digit_integer_eq_55_l1835_183536


namespace NUMINAMATH_GPT_largest_n_unique_k_l1835_183570

theorem largest_n_unique_k :
  ∃ (n : ℕ), (∀ (k1 k2 : ℕ), 
    (9 / 17 < n / (n + k1) → n / (n + k1) < 8 / 15 → 9 / 17 < n / (n + k2) → n / (n + k2) < 8 / 15 → k1 = k2) ∧ 
    n = 72) :=
sorry

end NUMINAMATH_GPT_largest_n_unique_k_l1835_183570


namespace NUMINAMATH_GPT_expression_is_integer_expression_modulo_3_l1835_183572

theorem expression_is_integer (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℤ), (n^3 + (3/2) * n^2 + (1/2) * n - 1) = k := 
sorry

theorem expression_modulo_3 (n : ℕ) (hn : n > 0) : 
  (n^3 + (3/2) * n^2 + (1/2) * n - 1) % 3 = 2 :=
sorry

end NUMINAMATH_GPT_expression_is_integer_expression_modulo_3_l1835_183572


namespace NUMINAMATH_GPT_tangent_neg_five_pi_six_eq_one_over_sqrt_three_l1835_183556

noncomputable def tangent_neg_five_pi_six : Real :=
  Real.tan (-5 * Real.pi / 6)

theorem tangent_neg_five_pi_six_eq_one_over_sqrt_three :
  tangent_neg_five_pi_six = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tangent_neg_five_pi_six_eq_one_over_sqrt_three_l1835_183556


namespace NUMINAMATH_GPT_percentage_decrease_last_year_l1835_183571

-- Define the percentage decrease last year
variable (x : ℝ)

-- Define the condition that expresses the stock price this year
def final_price_change (x : ℝ) : Prop :=
  (1 - x / 100) * 1.10 = 1 + 4.499999999999993 / 100

-- Theorem stating the percentage decrease
theorem percentage_decrease_last_year : final_price_change 5 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_last_year_l1835_183571


namespace NUMINAMATH_GPT_john_speed_l1835_183587

def johns_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ) : ℕ :=
    let john_time_min := next_fastest_guy_time_min - won_by_min
    let john_time_hr := john_time_min / 60
    race_distance_miles / john_time_hr

theorem john_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ)
    (h1 : race_distance_miles = 5) (h2 : next_fastest_guy_time_min = 23) (h3 : won_by_min = 3) : 
    johns_speed race_distance_miles next_fastest_guy_time_min won_by_min = 15 := 
by
    sorry

end NUMINAMATH_GPT_john_speed_l1835_183587


namespace NUMINAMATH_GPT_find_radius_of_circle_l1835_183513

theorem find_radius_of_circle (C : ℝ) (h : C = 72 * Real.pi) : ∃ r : ℝ, 2 * Real.pi * r = C ∧ r = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_of_circle_l1835_183513


namespace NUMINAMATH_GPT_find_base_k_l1835_183582

-- Define the conversion condition as a polynomial equation.
def base_conversion (k : ℤ) : Prop := k^2 + 3*k + 2 = 42

-- State the theorem to be proven: given the conversion condition, k = 5.
theorem find_base_k (k : ℤ) (h : base_conversion k) : k = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_base_k_l1835_183582


namespace NUMINAMATH_GPT_lucy_needs_more_distance_l1835_183502

noncomputable def mary_distance : ℝ := (3 / 8) * 24
noncomputable def edna_distance : ℝ := (2 / 3) * mary_distance
noncomputable def lucy_distance : ℝ := (5 / 6) * edna_distance

theorem lucy_needs_more_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end NUMINAMATH_GPT_lucy_needs_more_distance_l1835_183502


namespace NUMINAMATH_GPT_English_family_information_l1835_183541

-- Define the statements given by the family members.
variables (father_statement : Prop)
          (mother_statement : Prop)
          (daughter_statement : Prop)

-- Conditions provided in the problem
variables (going_to_Spain : Prop)
          (coming_from_Newcastle : Prop)
          (stopped_in_Paris : Prop)

-- Define what each family member said
axiom Father : father_statement ↔ (going_to_Spain ∨ coming_from_Newcastle)
axiom Mother : mother_statement ↔ ((¬going_to_Spain ∧ coming_from_Newcastle) ∨ (stopped_in_Paris ∧ ¬going_to_Spain))
axiom Daughter : daughter_statement ↔ (¬coming_from_Newcastle ∨ stopped_in_Paris)

-- The final theorem to be proved:
theorem English_family_information : (¬going_to_Spain ∧ coming_from_Newcastle ∧ stopped_in_Paris) :=
by
  -- steps to prove the theorem should go here, but they are skipped with sorry
  sorry

end NUMINAMATH_GPT_English_family_information_l1835_183541


namespace NUMINAMATH_GPT_double_root_polynomial_l1835_183575

theorem double_root_polynomial (b4 b3 b2 b1 : ℤ) (s : ℤ) :
  (Polynomial.eval s (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24) = 0)
  ∧ (Polynomial.eval s (Polynomial.derivative (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24)) = 0)
  → s = 1 ∨ s = -1 ∨ s = 2 ∨ s = -2 :=
by
  sorry

end NUMINAMATH_GPT_double_root_polynomial_l1835_183575


namespace NUMINAMATH_GPT_find_k_l1835_183598

theorem find_k : ∃ k : ℕ, 32 / k = 4 ∧ k = 8 := 
sorry

end NUMINAMATH_GPT_find_k_l1835_183598


namespace NUMINAMATH_GPT_pink_tulips_l1835_183532

theorem pink_tulips (total_tulips : ℕ)
    (blue_ratio : ℚ) (red_ratio : ℚ)
    (h_total : total_tulips = 56)
    (h_blue_ratio : blue_ratio = 3/8)
    (h_red_ratio : red_ratio = 3/7) :
    ∃ pink_tulips : ℕ, pink_tulips = total_tulips - ((blue_ratio * total_tulips) + (red_ratio * total_tulips)) ∧ pink_tulips = 11 := by
  sorry

end NUMINAMATH_GPT_pink_tulips_l1835_183532


namespace NUMINAMATH_GPT_coeff_x3_in_expansion_of_x_plus_1_50_l1835_183599

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end NUMINAMATH_GPT_coeff_x3_in_expansion_of_x_plus_1_50_l1835_183599


namespace NUMINAMATH_GPT_Bruce_paid_correct_amount_l1835_183504

def grape_kg := 9
def grape_price_per_kg := 70
def mango_kg := 7
def mango_price_per_kg := 55
def orange_kg := 5
def orange_price_per_kg := 45
def apple_kg := 3
def apple_price_per_kg := 80

def total_cost := grape_kg * grape_price_per_kg + 
                  mango_kg * mango_price_per_kg + 
                  orange_kg * orange_price_per_kg + 
                  apple_kg * apple_price_per_kg

theorem Bruce_paid_correct_amount : total_cost = 1480 := by
  sorry

end NUMINAMATH_GPT_Bruce_paid_correct_amount_l1835_183504


namespace NUMINAMATH_GPT_maxwell_age_l1835_183509

theorem maxwell_age (M : ℕ) (h1 : ∃ n : ℕ, n = M + 2) (h2 : ∃ k : ℕ, k = 4) (h3 : (M + 2) = 2 * 4) : M = 6 :=
sorry

end NUMINAMATH_GPT_maxwell_age_l1835_183509


namespace NUMINAMATH_GPT_inequality_condition_l1835_183546

theorem inequality_condition
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 :=
by
  sorry

end NUMINAMATH_GPT_inequality_condition_l1835_183546


namespace NUMINAMATH_GPT_value_of_x_l1835_183501

theorem value_of_x : 
  ∀ (x : ℕ), x = (2011^2 + 2011) / 2011 → x = 2012 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_value_of_x_l1835_183501


namespace NUMINAMATH_GPT_negative_correction_is_correct_l1835_183515

-- Define the constants given in the problem
def gain_per_day : ℚ := 13 / 4
def set_time : ℚ := 8 -- 8 A.M. on April 10
def end_time : ℚ := 15 -- 3 P.M. on April 19
def days_passed : ℚ := 9

-- Calculate the total time in hours from 8 A.M. on April 10 to 3 P.M. on April 19
def total_hours_passed : ℚ := days_passed * 24 + (end_time - set_time)

-- Calculate the gain in time per hour
def gain_per_hour : ℚ := gain_per_day / 24

-- Calculate the total gained time over the total hours passed
def total_gain : ℚ := total_hours_passed * gain_per_hour

-- The negative correction m to be subtracted
def correction : ℚ := 2899 / 96

theorem negative_correction_is_correct :
  total_gain = correction :=
by
-- skipping the proof
sorry

end NUMINAMATH_GPT_negative_correction_is_correct_l1835_183515


namespace NUMINAMATH_GPT_number_of_valid_sets_l1835_183585

open Set

variable {α : Type} (a b : α)

def is_valid_set (M : Set α) : Prop := M ∪ {a} = {a, b}

theorem number_of_valid_sets (a b : α) : (∃! M : Set α, is_valid_set a b M) := 
sorry

end NUMINAMATH_GPT_number_of_valid_sets_l1835_183585


namespace NUMINAMATH_GPT_find_roots_l1835_183573

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_find_roots_l1835_183573


namespace NUMINAMATH_GPT_roses_per_flat_l1835_183545

-- Conditions
def flats_petunias := 4
def petunias_per_flat := 8
def flats_roses := 3
def venus_flytraps := 2
def fertilizer_per_petunia := 8
def fertilizer_per_rose := 3
def fertilizer_per_venus_flytrap := 2
def total_fertilizer_needed := 314

-- Derived definitions
def total_petunias := flats_petunias * petunias_per_flat
def fertilizer_for_petunias := total_petunias * fertilizer_per_petunia
def fertilizer_for_venus_flytraps := venus_flytraps * fertilizer_per_venus_flytrap
def total_fertilizer_needed_roses := total_fertilizer_needed - (fertilizer_for_petunias + fertilizer_for_venus_flytraps)

-- Proof statement
theorem roses_per_flat :
  ∃ R : ℕ, flats_roses * R * fertilizer_per_rose = total_fertilizer_needed_roses ∧ R = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_roses_per_flat_l1835_183545


namespace NUMINAMATH_GPT_average_class_size_l1835_183567

theorem average_class_size 
  (three_year_olds : ℕ := 13)
  (four_year_olds : ℕ := 20)
  (five_year_olds : ℕ := 15)
  (six_year_olds : ℕ := 22) : 
  ((three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2) = 35 := 
by
  sorry

end NUMINAMATH_GPT_average_class_size_l1835_183567


namespace NUMINAMATH_GPT_dodecahedron_diagonals_l1835_183555

-- Define a structure representing a dodecahedron with its properties
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_meeting_at_each_vertex : Nat

-- Concretely define a dodecahedron based on the given problem properties
def dodecahedron_example : Dodecahedron :=
  { faces := 12,
    vertices := 20,
    faces_meeting_at_each_vertex := 3 }

-- Lean statement to prove the number of interior diagonals in a dodecahedron
theorem dodecahedron_diagonals (d : Dodecahedron) (h : d = dodecahedron_example) : 
  (d.vertices * (d.vertices - d.faces_meeting_at_each_vertex) / 2) = 160 := by
  rw [h]
  -- Even though we skip the proof, Lean should recognize the transformation
  sorry

end NUMINAMATH_GPT_dodecahedron_diagonals_l1835_183555


namespace NUMINAMATH_GPT_prime_division_or_divisibility_l1835_183558

open Nat

theorem prime_division_or_divisibility (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hodd : Odd p) (hd : p ∣ q^r + 1) :
    (2 * r ∣ p - 1) ∨ (p ∣ q^2 - 1) := 
sorry

end NUMINAMATH_GPT_prime_division_or_divisibility_l1835_183558


namespace NUMINAMATH_GPT_cos_E_floor_1000_l1835_183577

theorem cos_E_floor_1000 {EF GH FG EH : ℝ} {E G : ℝ} (h1 : EF = 200) (h2 : GH = 200) (h3 : FG + EH = 380) (h4 : E = G) (h5 : EH ≠ FG) :
  ∃ (cE : ℝ), cE = 11/16 ∧ ⌊ 1000 * cE ⌋ = 687 :=
by sorry

end NUMINAMATH_GPT_cos_E_floor_1000_l1835_183577


namespace NUMINAMATH_GPT_weight_of_oil_per_ml_l1835_183528

variable (w : ℝ)  -- Weight of the oil per ml
variable (total_volume : ℝ := 150)  -- Bowl volume
variable (oil_fraction : ℝ := 2/3)  -- Fraction of oil
variable (vinegar_fraction : ℝ := 1/3)  -- Fraction of vinegar
variable (vinegar_density : ℝ := 4)  -- Vinegar density in g/ml
variable (total_weight : ℝ := 700)  -- Total weight in grams

theorem weight_of_oil_per_ml :
  (total_volume * oil_fraction * w) + (total_volume * vinegar_fraction * vinegar_density) = total_weight →
  w = 5 := by
  sorry

end NUMINAMATH_GPT_weight_of_oil_per_ml_l1835_183528


namespace NUMINAMATH_GPT_abs_opposite_sign_eq_sum_l1835_183578

theorem abs_opposite_sign_eq_sum (a b : ℤ) (h : (|a + 1| * |b + 2| < 0)) : a + b = -3 :=
sorry

end NUMINAMATH_GPT_abs_opposite_sign_eq_sum_l1835_183578


namespace NUMINAMATH_GPT_solve_for_x_l1835_183535

-- Define the variables and conditions based on the problem statement
def equation (x : ℚ) := 5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)

-- State the theorem to be proved, including the condition and the result
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = 44.72727272727273 := by
  sorry  -- The proof is omitted

end NUMINAMATH_GPT_solve_for_x_l1835_183535


namespace NUMINAMATH_GPT_log_base_9_of_x_cubed_is_3_l1835_183506

theorem log_base_9_of_x_cubed_is_3 
  (x : Real) 
  (hx : x = 9.000000000000002) : 
  Real.logb 9 (x^3) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_log_base_9_of_x_cubed_is_3_l1835_183506


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1835_183518

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 500) (hS : S = 2500) (h_series : S = a / (1 - r)) : r = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1835_183518


namespace NUMINAMATH_GPT_tan_alpha_sol_expr_sol_l1835_183523

noncomputable def tan_half_alpha (α : ℝ) : ℝ := 2

noncomputable def tan_alpha_from_half (α : ℝ) : ℝ := 
  let tan_half := tan_half_alpha α
  2 * tan_half / (1 - tan_half * tan_half)

theorem tan_alpha_sol (α : ℝ) (h : tan_half_alpha α = 2) : tan_alpha_from_half α = -4 / 3 := by
  sorry

noncomputable def expr_eval (α : ℝ) : ℝ :=
  let tan_α := tan_alpha_from_half α
  let sin_α := tan_α / Real.sqrt (1 + tan_α * tan_α)
  let cos_α := 1 / Real.sqrt (1 + tan_α * tan_α)
  (6 * sin_α + cos_α) / (3 * sin_α - 2 * cos_α)

theorem expr_sol (α : ℝ) (h : tan_half_alpha α = 2) : expr_eval α = 7 / 6 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_sol_expr_sol_l1835_183523


namespace NUMINAMATH_GPT_pigeonhole_6_points_3x4_l1835_183592

theorem pigeonhole_6_points_3x4 :
  ∀ (points : Fin 6 → (ℝ × ℝ)), 
  (∀ i, 0 ≤ (points i).fst ∧ (points i).fst ≤ 4 ∧ 0 ≤ (points i).snd ∧ (points i).snd ≤ 3) →
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_pigeonhole_6_points_3x4_l1835_183592


namespace NUMINAMATH_GPT_sum_of_largest_two_l1835_183586

-- Define the three numbers
def a := 10
def b := 11
def c := 12

-- Define the sum of the largest and the next largest numbers
def sum_of_largest_two_numbers (x y z : ℕ) : ℕ :=
  if x >= y ∧ y >= z then x + y
  else if x >= z ∧ z >= y then x + z
  else if y >= x ∧ x >= z then y + x
  else if y >= z ∧ z >= x then y + z
  else if z >= x ∧ x >= y then z + x
  else z + y

-- State the theorem to prove
theorem sum_of_largest_two (x y z : ℕ) : sum_of_largest_two_numbers x y z = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_largest_two_l1835_183586


namespace NUMINAMATH_GPT_domain_of_f_3x_minus_1_domain_of_f_l1835_183525

-- Problem (1): Domain of f(3x - 1)
theorem domain_of_f_3x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ f x ∧ f x ≤ 1) →
  (∀ x, -1 / 3 ≤ x ∧ x ≤ 2 / 3) :=
by
  intro h
  sorry

-- Problem (2): Domain of f(x)
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, -1 ≤ 2*x + 5 ∧ 2*x + 5 ≤ 4) →
  (∀ y, 3 ≤ y ∧ y ≤ 13) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_domain_of_f_3x_minus_1_domain_of_f_l1835_183525


namespace NUMINAMATH_GPT_average_of_11_results_l1835_183553

theorem average_of_11_results 
  (S1: ℝ) (S2: ℝ) (fifth_result: ℝ) -- Define the variables
  (h1: S1 / 5 = 49)                -- sum of the first 5 results
  (h2: S2 / 7 = 52)                -- sum of the last 7 results
  (h3: fifth_result = 147)         -- the fifth result 
  : (S1 + S2 - fifth_result) / 11 = 42 := -- statement of the problem
by
  sorry

end NUMINAMATH_GPT_average_of_11_results_l1835_183553


namespace NUMINAMATH_GPT_part1_part2_l1835_183569

variable (α : Real)
-- Condition
axiom tan_neg_alpha : Real.tan (-α) = -2

-- Question 1
theorem part1 : ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α)) = 3 := 
by
  sorry

-- Question 2
theorem part2 : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1835_183569


namespace NUMINAMATH_GPT_find_sisters_dolls_l1835_183540

variable (H S : ℕ)

-- Conditions
def hannah_has_5_times_sisters_dolls : Prop :=
  H = 5 * S

def total_dolls_is_48 : Prop :=
  H + S = 48

-- Question: Prove S = 8
theorem find_sisters_dolls (h1 : hannah_has_5_times_sisters_dolls H S) (h2 : total_dolls_is_48 H S) : S = 8 :=
sorry

end NUMINAMATH_GPT_find_sisters_dolls_l1835_183540


namespace NUMINAMATH_GPT_computation_result_l1835_183559

theorem computation_result : 8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end NUMINAMATH_GPT_computation_result_l1835_183559


namespace NUMINAMATH_GPT_steve_fraction_of_day_in_school_l1835_183510

theorem steve_fraction_of_day_in_school :
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  (school_hours / total_hours) = (1 / 6) :=
by
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  have : (school_hours / total_hours) = (1 / 6) := sorry
  exact this

end NUMINAMATH_GPT_steve_fraction_of_day_in_school_l1835_183510


namespace NUMINAMATH_GPT_inequality_order_l1835_183574

theorem inequality_order (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) : 
  b > (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) > (a + b) / 2 ∧ (a + b) / 2 > 2 * a * b :=
by 
  sorry

end NUMINAMATH_GPT_inequality_order_l1835_183574


namespace NUMINAMATH_GPT_seeds_in_each_flower_bed_l1835_183565

theorem seeds_in_each_flower_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 54) (h2 : flower_beds = 9) : total_seeds / flower_beds = 6 :=
by
  sorry

end NUMINAMATH_GPT_seeds_in_each_flower_bed_l1835_183565


namespace NUMINAMATH_GPT_random_event_is_eventA_l1835_183564

-- Definitions of conditions
def eventA : Prop := true  -- Tossing a coin and it lands either heads up or tails up is a random event
def eventB : Prop := (∀ (a b : ℝ), (b * a = b * a))  -- The area of a rectangle with sides of length a and b is ab is a certain event
def eventC : Prop := ∃ (defective_items : ℕ), (defective_items / 100 = 10 / 100)  -- Drawing 2 defective items from 100 parts with 10% defective parts is uncertain
def eventD : Prop := false -- Scoring 105 points in a regular 100-point system exam is an impossible event

-- The proof problem statement
theorem random_event_is_eventA : eventA ∧ ¬eventB ∧ ¬eventC ∧ ¬eventD := 
sorry

end NUMINAMATH_GPT_random_event_is_eventA_l1835_183564


namespace NUMINAMATH_GPT_parabola_equation_l1835_183562

theorem parabola_equation (p : ℝ) (h : 0 < p) (Fₓ : ℝ) (Tₓ Tᵧ : ℝ) (Mₓ Mᵧ : ℝ)
  (eq_parabola : ∀ (y x : ℝ), y^2 = 2 * p * x → (y, x) = (Tᵧ, Tₓ))
  (F : (Fₓ, 0) = (p / 2, 0))
  (T_on_C : (Tᵧ, Tₓ) ∈ {(y, x) | y^2 = 2 * p * x})
  (FT_dist : dist (Fₓ, 0) (Tₓ, Tᵧ) = 5 / 2)
  (M : (Mₓ, Mᵧ) = (0, 1))
  (MF_MT_perp : ((Mᵧ - 0) / (Mₓ - Fₓ)) * ((Tᵧ - Mᵧ) / (Tₓ - Mᵧ)) = -1) :
  y^2 = 2 * x ∨ y^2 = 8 * x := 
sorry

end NUMINAMATH_GPT_parabola_equation_l1835_183562


namespace NUMINAMATH_GPT_eval_expression_l1835_183534

theorem eval_expression : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1835_183534


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l1835_183517

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9 / 13) (h2 : x - y = 5 / 13) : x^2 - y^2 = 45 / 169 := 
by 
  -- proof omitted 
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l1835_183517


namespace NUMINAMATH_GPT_point_on_number_line_l1835_183531

theorem point_on_number_line (a : ℤ) (h : abs (a + 3) = 4) : a = 1 ∨ a = -7 := 
sorry

end NUMINAMATH_GPT_point_on_number_line_l1835_183531


namespace NUMINAMATH_GPT_tangent_line_eqn_extreme_values_l1835_183503

/-- The tangent line to the function f at (0, 5) -/
theorem tangent_line_eqn (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ k b, (∀ x, f x = k * x + b) ∧ k = -2 ∧ b = 5) ∧ (2 * 0 + 5 - 5 = 0) := by
  sorry

/-- The function f has a local maximum at x = -1 and a local minimum at x = 2 -/
theorem extreme_values (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ x₁ x₂, x₁ = -1 ∧ f x₁ = 37 / 6 ∧ x₂ = 2 ∧ f x₂ = 5 / 3) := by
  sorry

end NUMINAMATH_GPT_tangent_line_eqn_extreme_values_l1835_183503


namespace NUMINAMATH_GPT_find_x_l1835_183533

theorem find_x (x : ℕ) : (x % 7 = 0) ∧ (x^2 > 200) ∧ (x < 30) ↔ (x = 21 ∨ x = 28) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1835_183533


namespace NUMINAMATH_GPT_reading_time_difference_l1835_183589

theorem reading_time_difference :
  let xanthia_reading_speed := 100 -- pages per hour
  let molly_reading_speed := 50 -- pages per hour
  let book_pages := 225
  let xanthia_time := book_pages / xanthia_reading_speed
  let molly_time := book_pages / molly_reading_speed
  let difference_in_hours := molly_time - xanthia_time
  let difference_in_minutes := difference_in_hours * 60
  difference_in_minutes = 135 := by
  sorry

end NUMINAMATH_GPT_reading_time_difference_l1835_183589


namespace NUMINAMATH_GPT_number_of_cups_needed_to_fill_container_l1835_183584

theorem number_of_cups_needed_to_fill_container (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 640) (h2 : cup_capacity = 120) : 
  (container_capacity + cup_capacity - 1) / cup_capacity = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cups_needed_to_fill_container_l1835_183584


namespace NUMINAMATH_GPT_find_angle_BEC_l1835_183539

-- Constants and assumptions
def angle_A : ℝ := 45
def angle_D : ℝ := 50
def angle_F : ℝ := 55
def E_above_C : Prop := true  -- This is a placeholder to represent the condition that E is directly above C.

-- Definition of the problem
theorem find_angle_BEC (angle_A_eq : angle_A = 45) 
                      (angle_D_eq : angle_D = 50) 
                      (angle_F_eq : angle_F = 55)
                      (triangle_BEC_formed : Prop)
                      (E_directly_above_C : E_above_C) 
                      : ∃ (BEC : ℝ), BEC = 10 :=
by sorry

end NUMINAMATH_GPT_find_angle_BEC_l1835_183539


namespace NUMINAMATH_GPT_difference_of_numbers_l1835_183508

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 12390) (h2 : b = 2 * a + 18) : b - a = 4142 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_of_numbers_l1835_183508


namespace NUMINAMATH_GPT_first_train_cross_time_l1835_183505

noncomputable def length_first_train : ℝ := 800
noncomputable def speed_first_train_kmph : ℝ := 120
noncomputable def length_second_train : ℝ := 1000
noncomputable def speed_second_train_kmph : ℝ := 80
noncomputable def length_third_train : ℝ := 600
noncomputable def speed_third_train_kmph : ℝ := 150

noncomputable def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

noncomputable def speed_first_train_mps : ℝ := speed_kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train_mps : ℝ := speed_kmph_to_mps speed_second_train_kmph
noncomputable def speed_third_train_mps : ℝ := speed_kmph_to_mps speed_third_train_kmph

noncomputable def relative_speed_same_direction : ℝ := speed_first_train_mps - speed_second_train_mps
noncomputable def relative_speed_opposite_direction : ℝ := speed_first_train_mps + speed_third_train_mps

noncomputable def time_to_cross_second_train : ℝ := (length_first_train + length_second_train) / relative_speed_same_direction
noncomputable def time_to_cross_third_train : ℝ := (length_first_train + length_third_train) / relative_speed_opposite_direction

noncomputable def total_time_to_cross : ℝ := time_to_cross_second_train + time_to_cross_third_train

theorem first_train_cross_time : total_time_to_cross = 180.67 := by
  sorry

end NUMINAMATH_GPT_first_train_cross_time_l1835_183505


namespace NUMINAMATH_GPT_expected_number_of_different_faces_l1835_183591

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end NUMINAMATH_GPT_expected_number_of_different_faces_l1835_183591


namespace NUMINAMATH_GPT_probability_of_five_dice_all_same_l1835_183595

theorem probability_of_five_dice_all_same : 
  (6 / (6 ^ 5) = 1 / 1296) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_five_dice_all_same_l1835_183595


namespace NUMINAMATH_GPT_consecutive_odd_integers_l1835_183522

theorem consecutive_odd_integers (n : ℤ) (h : (n - 2) + (n + 2) = 130) : n = 65 :=
sorry

end NUMINAMATH_GPT_consecutive_odd_integers_l1835_183522


namespace NUMINAMATH_GPT_eugene_total_cost_l1835_183529

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end NUMINAMATH_GPT_eugene_total_cost_l1835_183529


namespace NUMINAMATH_GPT_no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l1835_183550

theorem no_natural_n_such_that_6n2_plus_5n_is_power_of_2 :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 6 * n^2 + 5 * n = 2^k :=
by
  sorry

end NUMINAMATH_GPT_no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l1835_183550


namespace NUMINAMATH_GPT_maximum_guaranteed_money_l1835_183581

theorem maximum_guaranteed_money (board_width board_height tromino_width tromino_height guaranteed_rubles : ℕ) 
  (h_board_width : board_width = 21) 
  (h_board_height : board_height = 20)
  (h_tromino_width : tromino_width = 3) 
  (h_tromino_height : tromino_height = 1)
  (h_guaranteed_rubles : guaranteed_rubles = 14) :
  true := by
  sorry

end NUMINAMATH_GPT_maximum_guaranteed_money_l1835_183581


namespace NUMINAMATH_GPT_find_positive_integer_l1835_183538

theorem find_positive_integer (n : ℕ) (h1 : 100 % n = 3) (h2 : 197 % n = 3) : n = 97 := 
sorry

end NUMINAMATH_GPT_find_positive_integer_l1835_183538


namespace NUMINAMATH_GPT_circle_radius_l1835_183583

-- Define the general equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Prove the radius of the circle given by the equation is √5
theorem circle_radius :
  (∀ x y : ℝ, circle_eq x y) →
  (∃ r : ℝ, r = Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1835_183583


namespace NUMINAMATH_GPT_find_a_l1835_183563

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (4254253 % 53^1 - a) % 17 = 0): 
  a = 3 := 
sorry

end NUMINAMATH_GPT_find_a_l1835_183563


namespace NUMINAMATH_GPT_min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l1835_183549

namespace MathProof

-- Definitions and conditions
variables {x y : ℝ}
axiom x_pos : x > 0
axiom y_pos : y > 0
axiom sum_eq_one : x + y = 1

-- Problem Statement 1: Prove the minimum value of x^2 + y^2 is 1/2
theorem min_value_of_x2_plus_y2 : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ (x^2 + y^2 = 1/2) :=
by
  sorry

-- Problem Statement 2: Prove the minimum value of 1/x + 1/y + 1/(xy) is 6
theorem min_value_of_reciprocal_sum : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ ((1/x + 1/y + 1/(x*y)) = 6) :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l1835_183549


namespace NUMINAMATH_GPT_thirteen_percent_greater_than_80_l1835_183507

theorem thirteen_percent_greater_than_80 (x : ℝ) (h : x = 1.13 * 80) : x = 90.4 :=
sorry

end NUMINAMATH_GPT_thirteen_percent_greater_than_80_l1835_183507


namespace NUMINAMATH_GPT_business_value_l1835_183560

-- Define the conditions
variable (V : ℝ) -- Total value of the business
variable (man_shares : ℝ := (2/3) * V) -- Man's share in the business
variable (sold_shares_value : ℝ := (3/4) * man_shares) -- Value of sold shares
variable (sale_price : ℝ := 45000) -- Price the shares were sold for

-- State the theorem to be proven
theorem business_value (h : (3/4) * (2/3) * V = 45000) : V = 90000 := by
  sorry

end NUMINAMATH_GPT_business_value_l1835_183560


namespace NUMINAMATH_GPT_min_games_required_l1835_183576

-- Given condition: max_games ≤ 15
def max_games := 15

-- Theorem statement to prove: minimum number of games that must be played is 8
theorem min_games_required (n : ℕ) (h : n ≤ max_games) : n = 8 :=
sorry

end NUMINAMATH_GPT_min_games_required_l1835_183576


namespace NUMINAMATH_GPT_students_neither_cs_nor_robotics_l1835_183597

theorem students_neither_cs_nor_robotics
  (total_students : ℕ)
  (cs_students : ℕ)
  (robotics_students : ℕ)
  (both_cs_and_robotics : ℕ)
  (H1 : total_students = 150)
  (H2 : cs_students = 90)
  (H3 : robotics_students = 70)
  (H4 : both_cs_and_robotics = 20) :
  (total_students - (cs_students + robotics_students - both_cs_and_robotics)) = 10 :=
by
  sorry

end NUMINAMATH_GPT_students_neither_cs_nor_robotics_l1835_183597


namespace NUMINAMATH_GPT_john_receives_more_l1835_183511

noncomputable def partnership_difference (investment_john : ℝ) (investment_mike : ℝ) (profit : ℝ) : ℝ :=
  let total_investment := investment_john + investment_mike
  let one_third_profit := profit / 3
  let two_third_profit := 2 * profit / 3
  let john_effort_share := one_third_profit / 2
  let mike_effort_share := one_third_profit / 2
  let ratio_john := investment_john / total_investment
  let ratio_mike := investment_mike / total_investment
  let john_investment_share := ratio_john * two_third_profit
  let mike_investment_share := ratio_mike * two_third_profit
  let john_total := john_effort_share + john_investment_share
  let mike_total := mike_effort_share + mike_investment_share
  john_total - mike_total

theorem john_receives_more (investment_john investment_mike profit : ℝ)
  (h_john : investment_john = 700)
  (h_mike : investment_mike = 300)
  (h_profit : profit = 3000.0000000000005) :
  partnership_difference investment_john investment_mike profit = 800.0000000000001 := 
sorry

end NUMINAMATH_GPT_john_receives_more_l1835_183511


namespace NUMINAMATH_GPT_lines_parallel_iff_a_eq_1_l1835_183580

theorem lines_parallel_iff_a_eq_1 (x y a : ℝ) :
    (a = 1 ↔ ∃ k : ℝ, ∀ x y : ℝ, a*x + y - 1 = k*(x + a*y + 1)) :=
sorry

end NUMINAMATH_GPT_lines_parallel_iff_a_eq_1_l1835_183580


namespace NUMINAMATH_GPT_quadratic_function_passing_origin_l1835_183514

theorem quadratic_function_passing_origin (a : ℝ) (h : ∃ x y, y = ax^2 + x + a * (a - 2) ∧ (x, y) = (0, 0)) : a = 2 := by
  sorry

end NUMINAMATH_GPT_quadratic_function_passing_origin_l1835_183514


namespace NUMINAMATH_GPT_proof_l1835_183512

noncomputable def problem : Prop :=
  let a := 1
  let b := 2
  let angleC := 60 * Real.pi / 180 -- convert degrees to radians
  let cosC := Real.cos angleC
  let sinC := Real.sin angleC
  let c_squared := a^2 + b^2 - 2 * a * b * cosC
  let c := Real.sqrt c_squared
  let area := 0.5 * a * b * sinC
  c = Real.sqrt 3 ∧ area = Real.sqrt 3 / 2

theorem proof : problem :=
by
  sorry

end NUMINAMATH_GPT_proof_l1835_183512


namespace NUMINAMATH_GPT_total_oranges_over_four_days_l1835_183590

def jeremy_oranges_monday := 100
def jeremy_oranges_tuesday (B: ℕ) := 3 * jeremy_oranges_monday
def jeremy_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B)
def jeremy_oranges_thursday := 70
def brother_oranges_tuesday := 3 * jeremy_oranges_monday - jeremy_oranges_monday -- This is B from Tuesday
def cousin_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B) - (jeremy_oranges_monday + B)

theorem total_oranges_over_four_days (B: ℕ) (C: ℕ)
        (B_equals_tuesday: B = brother_oranges_tuesday)
        (J_plus_B_equals_300 : jeremy_oranges_tuesday B = 300)
        (J_plus_B_plus_C_equals_600 : jeremy_oranges_wednesday B C = 600)
        (J_thursday_is_70 : jeremy_oranges_thursday = 70)
        (B_thursday_is_B : B = brother_oranges_tuesday):
    100 + 300 + 600 + 270 = 1270 := by
        sorry

end NUMINAMATH_GPT_total_oranges_over_four_days_l1835_183590


namespace NUMINAMATH_GPT_speed_in_km_per_hr_l1835_183566

noncomputable def side : ℝ := 40
noncomputable def time : ℝ := 64

-- Theorem statement
theorem speed_in_km_per_hr (side : ℝ) (time : ℝ) (h₁ : side = 40) (h₂ : time = 64) : 
  (4 * side * 3600) / (time * 1000) = 9 := by
  rw [h₁, h₂]
  sorry

end NUMINAMATH_GPT_speed_in_km_per_hr_l1835_183566


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1835_183593

theorem value_of_x_plus_y (x y : ℤ) (hx : x = -3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1835_183593


namespace NUMINAMATH_GPT_amy_required_hours_per_week_l1835_183547

variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_pay : ℕ) 
variable (pay_raise_percent : ℕ) (school_year_weeks : ℕ) (required_school_year_pay : ℕ)

def summer_hours_total := summer_hours_per_week * summer_weeks
def summer_hourly_pay := summer_pay / summer_hours_total
def new_hourly_pay := summer_hourly_pay + (summer_hourly_pay / 10)  -- 10% pay raise
def total_needed_hours := required_school_year_pay / new_hourly_pay
def required_hours_per_week := total_needed_hours / school_year_weeks

theorem amy_required_hours_per_week :
  summer_hours_per_week = 40 →
  summer_weeks = 12 →
  summer_pay = 4800 →
  pay_raise_percent = 10 →
  school_year_weeks = 36 →
  required_school_year_pay = 7200 →
  required_hours_per_week = 18 := sorry

end NUMINAMATH_GPT_amy_required_hours_per_week_l1835_183547


namespace NUMINAMATH_GPT_find_value_l1835_183543

variable (a b c : Int)

-- Conditions from the problem
axiom abs_a_eq_two : |a| = 2
axiom b_eq_neg_seven : b = -7
axiom neg_c_eq_neg_five : -c = -5

-- Proof problem
theorem find_value : a^2 + (-b) + (-c) = 6 := by
  sorry

end NUMINAMATH_GPT_find_value_l1835_183543


namespace NUMINAMATH_GPT_tangent_line_circle_l1835_183537

theorem tangent_line_circle (a : ℝ) :
  (∀ (x y : ℝ), 4 * x - 3 * y = 0 → x^2 + y^2 - 2 * x + a * y + 1 = 0) →
  a = -1 ∨ a = 4 :=
sorry

end NUMINAMATH_GPT_tangent_line_circle_l1835_183537
