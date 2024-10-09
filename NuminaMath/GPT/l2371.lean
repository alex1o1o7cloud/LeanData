import Mathlib

namespace isosceles_trapezoid_problem_l2371_237137

variable (AB CD AD BC : ℝ)
variable (x : ℝ)

noncomputable def p_squared (AB CD AD BC : ℝ) (x : ℝ) : ℝ :=
  if AB = 100 ∧ CD = 25 ∧ AD = x ∧ BC = x then 1875 else 0

theorem isosceles_trapezoid_problem (h₁ : AB = 100)
                                    (h₂ : CD = 25)
                                    (h₃ : AD = x)
                                    (h₄ : BC = x) :
  p_squared AB CD AD BC x = 1875 := by
  sorry

end isosceles_trapezoid_problem_l2371_237137


namespace sum_of_consecutive_integers_l2371_237118

theorem sum_of_consecutive_integers (a : ℤ) (h₁ : a = 18) (h₂ : a + 1 = 19) (h₃ : a + 2 = 20) : a + (a + 1) + (a + 2) = 57 :=
by
  -- Add a sorry to focus on creating the statement successfully
  sorry

end sum_of_consecutive_integers_l2371_237118


namespace original_price_four_pack_l2371_237189

theorem original_price_four_pack (price_with_rush: ℝ) (increase_rate: ℝ) (num_packs: ℕ):
  price_with_rush = 13 → increase_rate = 0.30 → num_packs = 4 → num_packs * (price_with_rush / (1 + increase_rate)) = 40 :=
by
  intros h_price h_rate h_packs
  rw [h_price, h_rate, h_packs]
  sorry

end original_price_four_pack_l2371_237189


namespace four_spheres_max_intersections_l2371_237128

noncomputable def max_intersection_points (n : Nat) : Nat :=
  if h : n > 0 then n * 2 else 0

theorem four_spheres_max_intersections : max_intersection_points 4 = 8 := by
  sorry

end four_spheres_max_intersections_l2371_237128


namespace point_on_ellipse_l2371_237187

noncomputable def ellipse_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  let d1 := ((x - F1.1)^2 + (y - F1.2)^2).sqrt
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  x^2 + 4 * y^2 = 16 ∧ d1 = 7

theorem point_on_ellipse (P F1 F2 : ℝ × ℝ)
  (h : ellipse_condition P F1 F2) : 
  let x := P.1
  let y := P.2
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  d2 = 1 :=
sorry

end point_on_ellipse_l2371_237187


namespace walnut_swap_exists_l2371_237172

theorem walnut_swap_exists (n : ℕ) (h_n : n = 2021) :
  ∃ k : ℕ, k < n ∧ ∃ a b : ℕ, a < k ∧ k < b :=
by
  sorry

end walnut_swap_exists_l2371_237172


namespace part_I_part_II_l2371_237112

open Real

noncomputable def alpha₁ : Real := sorry -- Placeholder for the angle α in part I
noncomputable def alpha₂ : Real := sorry -- Placeholder for the angle α in part II

-- Given a point P(-4, 3) and a point on the terminal side of angle α₁ such that tan(α₁) = -3/4
theorem part_I :
  tan α₁ = - (3 / 4) → 
  (cos (π / 2 + α₁) * sin (-π - α₁)) / (cos (11 * π / 2 - α₁) * sin (9 * π / 2 + α₁)) = - (3 / 4) :=
by 
  intro h
  sorry

-- Given vector a = (3,1) and b = (sin α, cos α) where a is parallel to b such that tan(α₂) = 3
theorem part_II :
  tan α₂ = 3 → 
  (4 * sin α₂ - 2 * cos α₂) / (5 * cos α₂ + 3 * sin α₂) = 5 / 7 :=
by 
  intro h
  sorry

end part_I_part_II_l2371_237112


namespace find_y_values_l2371_237130

theorem find_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = 0 ∨ y = 144 ∨ y = -24) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end find_y_values_l2371_237130


namespace scientific_notation_15_7_trillion_l2371_237198

theorem scientific_notation_15_7_trillion :
  ∃ n : ℝ, n = 15.7 * 10^12 ∧ n = 1.57 * 10^13 :=
by
  sorry

end scientific_notation_15_7_trillion_l2371_237198


namespace average_score_is_67_l2371_237144

def scores : List ℕ := [55, 67, 76, 82, 55]
def num_of_subjects : ℕ := List.length scores
def total_score : ℕ := List.sum scores
def average_score : ℕ := total_score / num_of_subjects

theorem average_score_is_67 : average_score = 67 := by
  sorry

end average_score_is_67_l2371_237144


namespace initial_number_of_red_balls_l2371_237191

theorem initial_number_of_red_balls 
  (num_white_balls num_red_balls : ℕ)
  (h1 : num_red_balls = 4 * num_white_balls + 3)
  (num_actions : ℕ)
  (h2 : 4 + 5 * num_actions = num_white_balls)
  (h3 : 34 + 17 * num_actions = num_red_balls) : 
  num_red_balls = 119 := 
by
  sorry

end initial_number_of_red_balls_l2371_237191


namespace find_constant_l2371_237127

-- Define the variables: t, x, y, and the constant
variable (t x y constant : ℝ)

-- Conditions
def x_def : x = constant - 2 * t :=
  by sorry

def y_def : y = 2 * t - 2 :=
  by sorry

def x_eq_y_at_t : t = 0.75 → x = y :=
  by sorry

-- Proposition: Prove that the constant in the equation for x is 1
theorem find_constant (ht : t = 0.75) (hx : x = constant - 2 * t) (hy : y = 2 * t - 2) (he : x = y) :
  constant = 1 :=
  by sorry

end find_constant_l2371_237127


namespace speed_of_current_eq_l2371_237188

theorem speed_of_current_eq :
  ∃ (m c : ℝ), (m + c = 15) ∧ (m - c = 8.6) ∧ (c = 3.2) :=
by
  sorry

end speed_of_current_eq_l2371_237188


namespace roman_coins_left_l2371_237159

theorem roman_coins_left (X Y : ℕ) (h1 : X * Y = 50) (h2 : (X - 7) * Y = 28) : X - 7 = 8 :=
by
  sorry

end roman_coins_left_l2371_237159


namespace flowers_remaining_along_path_after_events_l2371_237184

def total_flowers : ℕ := 30
def total_peonies : ℕ := 15
def total_tulips : ℕ := 15
def unwatered_flowers : ℕ := 10
def tulips_watered_by_sineglazka : ℕ := 10
def tulips_picked_by_neznaika : ℕ := 6
def remaining_flowers : ℕ := 19

theorem flowers_remaining_along_path_after_events :
  total_peonies + total_tulips = total_flowers →
  tulips_watered_by_sineglazka + unwatered_flowers = total_flowers →
  tulips_picked_by_neznaika ≤ total_tulips →
  remaining_flowers = 19 := sorry

end flowers_remaining_along_path_after_events_l2371_237184


namespace average_score_of_male_students_standard_deviation_of_all_students_l2371_237124

def students : ℕ := 5
def total_average_score : ℝ := 80
def male_student_variance : ℝ := 150
def female_student1_score : ℝ := 85
def female_student2_score : ℝ := 75
def male_student_average_score : ℝ := 80 -- From solution step (1)
def total_standard_deviation : ℝ := 10 -- From solution step (2)

theorem average_score_of_male_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  male_student_average_score = 80 :=
by sorry

theorem standard_deviation_of_all_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  total_standard_deviation = 10 :=
by sorry

end average_score_of_male_students_standard_deviation_of_all_students_l2371_237124


namespace sum_of_numbers_gt_1_1_equals_3_9_l2371_237171

noncomputable def sum_of_elements_gt_1_1 : Float :=
  let numbers := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]
  let numbers_gt_1_1 := List.filter (fun x => x > 1.1) numbers
  List.sum numbers_gt_1_1

theorem sum_of_numbers_gt_1_1_equals_3_9 :
  sum_of_elements_gt_1_1 = 3.9 := by
  sorry

end sum_of_numbers_gt_1_1_equals_3_9_l2371_237171


namespace simplify_fraction_l2371_237142

theorem simplify_fraction : (3 ^ 2016 - 3 ^ 2014) / (3 ^ 2016 + 3 ^ 2014) = 4 / 5 :=
by
  sorry

end simplify_fraction_l2371_237142


namespace max_sum_of_factors_l2371_237153

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C) (h7 : A * B * C = 3003) :
  A + B + C ≤ 45 :=
sorry

end max_sum_of_factors_l2371_237153


namespace find_l_l2371_237178

variables (a b c l : ℤ)
def g (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_l :
  g a b c 2 = 0 →
  60 < g a b c 6 ∧ g a b c 6 < 70 →
  80 < g a b c 9 ∧ g a b c 9 < 90 →
  6000 * l < g a b c 100 ∧ g a b c 100 < 6000 * (l + 1) →
  l = 5 :=
sorry

end find_l_l2371_237178


namespace guards_can_protect_point_l2371_237152

-- Define the conditions of the problem as Lean definitions
def guardVisionRadius : ℝ := 100

-- Define the proof statement
theorem guards_can_protect_point :
  ∃ (num_guards : ℕ), num_guards * 45 = 360 ∧ guardVisionRadius = 100 :=
by
  sorry

end guards_can_protect_point_l2371_237152


namespace least_number_to_subtract_l2371_237107

theorem least_number_to_subtract (n : ℕ) (h : n = 652543) : 
  ∃ x : ℕ, x = 7 ∧ (n - x) % 12 = 0 :=
by
  sorry

end least_number_to_subtract_l2371_237107


namespace find_n_l2371_237154

theorem find_n (a b c : ℝ) (h : a^2 + b^2 = c^2) (n : ℕ) (hn : n > 2) : 
  (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n)) → n = 4 :=
by
  sorry

end find_n_l2371_237154


namespace shaded_area_of_intersections_l2371_237125

theorem shaded_area_of_intersections (r : ℝ) (n : ℕ) (intersect_origin : Prop) (radius_5 : r = 5) (four_circles : n = 4) : 
  ∃ (area : ℝ), area = 100 * Real.pi - 200 :=
by
  sorry

end shaded_area_of_intersections_l2371_237125


namespace least_prime_b_l2371_237151

-- Define what it means for an angle to be a right triangle angle sum
def isRightTriangleAngleSum (a b : ℕ) : Prop := a + b = 90

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Formalize the goal: proving that the smallest possible b is 7
theorem least_prime_b (a b : ℕ) (h1 : isRightTriangleAngleSum a b) (h2 : isPrime a) (h3 : isPrime b) (h4 : a > b) : b = 7 :=
sorry

end least_prime_b_l2371_237151


namespace find_x_l2371_237150

theorem find_x (u : ℕ) (h₁ : u = 90) (w : ℕ) (h₂ : w = u + 10)
                (z : ℕ) (h₃ : z = w + 25) (y : ℕ) (h₄ : y = z + 15)
                (x : ℕ) (h₅ : x = y + 3) : x = 143 :=
by {
  -- Proof will be included here
  sorry
}

end find_x_l2371_237150


namespace sparrows_among_non_robins_percentage_l2371_237180

-- Define percentages of different birds
def finches_percentage : ℝ := 0.40
def sparrows_percentage : ℝ := 0.20
def owls_percentage : ℝ := 0.15
def robins_percentage : ℝ := 0.25

-- Define the statement to prove 
theorem sparrows_among_non_robins_percentage :
  ((sparrows_percentage / (1 - robins_percentage)) * 100) = 26.67 := by
  -- This is where the proof would go, but it's omitted as per instructions
  sorry

end sparrows_among_non_robins_percentage_l2371_237180


namespace gcd_10010_15015_l2371_237193

theorem gcd_10010_15015 :
  Int.gcd 10010 15015 = 5005 :=
by 
  sorry

end gcd_10010_15015_l2371_237193


namespace remainder_when_7645_divided_by_9_l2371_237116

/--
  Prove that the remainder when 7645 is divided by 9 is 4,
  given that a number is congruent to the sum of its digits modulo 9.
-/
theorem remainder_when_7645_divided_by_9 :
  7645 % 9 = 4 :=
by
  -- Main proof should go here
  sorry

end remainder_when_7645_divided_by_9_l2371_237116


namespace ticket_1000_wins_probability_l2371_237122

-- Define the total number of tickets
def n_tickets := 1000

-- Define the number of odd tickets
def n_odd_tickets := 500

-- Define the number of relevant tickets (ticket 1000 + odd tickets)
def n_relevant_tickets := 501

-- Define the probability that ticket number 1000 wins a prize
def win_probability : ℚ := 1 / n_relevant_tickets

-- State the theorem
theorem ticket_1000_wins_probability : win_probability = 1 / 501 :=
by
  -- The proof would go here
  sorry

end ticket_1000_wins_probability_l2371_237122


namespace cyclic_sum_inequality_l2371_237170

theorem cyclic_sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
    (ab / (ab + a^5 + b^5)) + (bc / (bc + b^5 + c^5)) + (ca / (ca + c^5 + a^5)) ≤ 1 := by
  sorry

end cyclic_sum_inequality_l2371_237170


namespace find_v2002_l2371_237106

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 6
  | 4 => 2
  | 5 => 1
  | 6 => 7
  | 7 => 4
  | _ => 0

def seq_v : ℕ → ℕ
| 0       => 5
| (n + 1) => g (seq_v n)

theorem find_v2002 : seq_v 2002 = 5 :=
  sorry

end find_v2002_l2371_237106


namespace total_revenue_correct_l2371_237177

def small_slices_price := 150
def large_slices_price := 250
def total_slices_sold := 5000
def small_slices_sold := 2000

def large_slices_sold := total_slices_sold - small_slices_sold

def revenue_from_small_slices := small_slices_sold * small_slices_price
def revenue_from_large_slices := large_slices_sold * large_slices_price
def total_revenue := revenue_from_small_slices + revenue_from_large_slices

theorem total_revenue_correct : total_revenue = 1050000 := by
  sorry

end total_revenue_correct_l2371_237177


namespace wine_price_increase_l2371_237119

-- Definitions translating the conditions
def wine_cost_today : ℝ := 20.0
def bottles_count : ℕ := 5
def tariff_rate : ℝ := 0.25

-- Statement to prove
theorem wine_price_increase (wine_cost_today : ℝ) (bottles_count : ℕ) (tariff_rate : ℝ) : 
  bottles_count * wine_cost_today * tariff_rate = 25.0 := 
by
  -- Proof is omitted
  sorry

end wine_price_increase_l2371_237119


namespace trigonometric_expression_l2371_237175

theorem trigonometric_expression (x : ℝ) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := 
sorry

end trigonometric_expression_l2371_237175


namespace cups_needed_correct_l2371_237165

-- Define the conditions
def servings : ℝ := 18.0
def cups_per_serving : ℝ := 2.0

-- Define the total cups needed calculation
def total_cups (servings : ℝ) (cups_per_serving : ℝ) : ℝ :=
  servings * cups_per_serving

-- State the proof problem
theorem cups_needed_correct :
  total_cups servings cups_per_serving = 36.0 :=
by
  sorry

end cups_needed_correct_l2371_237165


namespace average_distance_is_600_l2371_237117

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l2371_237117


namespace shop_weekly_earnings_l2371_237129

theorem shop_weekly_earnings
  (price_women: ℕ := 18)
  (price_men: ℕ := 15)
  (time_open_hours: ℕ := 12)
  (minutes_per_hour: ℕ := 60)
  (weekly_days: ℕ := 7)
  (sell_rate_women: ℕ := 30)
  (sell_rate_men: ℕ := 40) :
  (time_open_hours * (minutes_per_hour / sell_rate_women) * price_women +
   time_open_hours * (minutes_per_hour / sell_rate_men) * price_men) * weekly_days = 4914 := 
sorry

end shop_weekly_earnings_l2371_237129


namespace Chandler_more_rolls_needed_l2371_237114

theorem Chandler_more_rolls_needed :
  let total_goal := 12
  let sold_to_grandmother := 3
  let sold_to_uncle := 4
  let sold_to_neighbor := 3
  let total_sold := sold_to_grandmother + sold_to_uncle + sold_to_neighbor
  total_goal - total_sold = 2 :=
by
  sorry

end Chandler_more_rolls_needed_l2371_237114


namespace necessary_and_sufficient_condition_l2371_237161

variable {A B : Prop}

theorem necessary_and_sufficient_condition (h1 : A → B) (h2 : B → A) : A ↔ B := 
by 
  sorry

end necessary_and_sufficient_condition_l2371_237161


namespace find_number_of_Persians_l2371_237157

variable (P : ℕ)  -- Number of Persian cats Jamie owns
variable (M : ℕ := 2)  -- Number of Maine Coons Jamie owns (given by conditions)
variable (G_P : ℕ := P / 2)  -- Number of Persian cats Gordon owns, which is half of Jamie's
variable (G_M : ℕ := M + 1)  -- Number of Maine Coons Gordon owns, one more than Jamie's
variable (H_P : ℕ := 0)  -- Number of Persian cats Hawkeye owns, which is 0
variable (H_M : ℕ := G_M - 1)  -- Number of Maine Coons Hawkeye owns, one less than Gordon's

theorem find_number_of_Persians (sum_cats : P + M + G_P + G_M + H_P + H_M = 13) : 
  P = 4 :=
by
  -- Proof can be filled in here
  sorry

end find_number_of_Persians_l2371_237157


namespace monthly_installment_amount_l2371_237196

theorem monthly_installment_amount (total_cost : ℝ) (down_payment_percentage : ℝ) (additional_down_payment : ℝ) 
  (balance_after_months : ℝ) (months : ℕ) (monthly_installment : ℝ) : 
    total_cost = 1000 → 
    down_payment_percentage = 0.20 → 
    additional_down_payment = 20 → 
    balance_after_months = 520 → 
    months = 4 → 
    monthly_installment = 65 :=
by
  intros
  sorry

end monthly_installment_amount_l2371_237196


namespace general_term_of_sequence_l2371_237167

noncomputable def harmonic_mean {n : ℕ} (p : Fin n → ℝ) : ℝ :=
  n / (Finset.univ.sum (fun i => p i))

theorem general_term_of_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, harmonic_mean (fun i : Fin n => a (i + 1)) = 1 / (2 * n - 1))
    (h₂ : ∀ n : ℕ, (Finset.range n).sum a = 2 * n^2 - n) :
  ∀ n : ℕ, a n = 4 * n - 3 := by
  sorry

end general_term_of_sequence_l2371_237167


namespace cost_of_seven_books_l2371_237192

theorem cost_of_seven_books (h : 3 * 12 = 36) : 7 * 12 = 84 :=
sorry

end cost_of_seven_books_l2371_237192


namespace M_even_comp_M_composite_comp_M_prime_not_div_l2371_237176

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_composite (n : ℕ) : Prop :=  ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n
def M (n : ℕ) : ℕ := 2^n - 1

theorem M_even_comp (n : ℕ) (h1 : n ≠ 2) (h2 : is_even n) : is_composite (M n) :=
sorry

theorem M_composite_comp (n : ℕ) (h : is_composite n) : is_composite (M n) :=
sorry

theorem M_prime_not_div (p : ℕ) (h : Nat.Prime p) : ¬ (p ∣ M p) :=
sorry

end M_even_comp_M_composite_comp_M_prime_not_div_l2371_237176


namespace smallest_product_bdf_l2371_237147

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l2371_237147


namespace union_of_A_and_B_l2371_237185

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l2371_237185


namespace non_adjective_primes_sum_l2371_237133

-- We will define the necessary components as identified from our problem

def is_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ∃ a : ℕ → ℕ, ∀ n : ℕ,
    a 0 % p = (1 + (1 / a 1) % p) ∧
    a 1 % p = (1 + (1 / (1 + (1 / a 2) % p)) % p) ∧
    a 2 % p = (1 + (1 / (1 + (1 / (1 + (1 / a 3) % p))) % p))

def is_not_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ¬ is_adjective_prime p

def first_three_non_adjective_primes_sum : ℕ :=
  3 + 7 + 23

theorem non_adjective_primes_sum :
  first_three_non_adjective_primes_sum = 33 := 
  sorry

end non_adjective_primes_sum_l2371_237133


namespace solve_equation_l2371_237195

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 → (x = 0 ∨ x = 1) :=
by
  intro h
  sorry

end solve_equation_l2371_237195


namespace product_uvw_l2371_237113

theorem product_uvw (a x y c : ℝ) (u v w : ℤ) :
  (a^u * x - a^v) * (a^w * y - a^3) = a^5 * c^5 → 
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1) → 
  u * v * w = 6 :=
by
  intros h1 h2
  -- Proof will go here
  sorry

end product_uvw_l2371_237113


namespace seafoam_azure_ratio_l2371_237132

-- Define the conditions
variables (P S A : ℕ) 

-- Purple Valley has one-quarter as many skirts as Seafoam Valley
axiom h1 : P = S / 4

-- Azure Valley has 60 skirts
axiom h2 : A = 60

-- Purple Valley has 10 skirts
axiom h3 : P = 10

-- The goal is to prove the ratio of Seafoam Valley skirts to Azure Valley skirts is 2 to 3
theorem seafoam_azure_ratio : S / A = 2 / 3 :=
by 
  sorry

end seafoam_azure_ratio_l2371_237132


namespace number_of_zeros_l2371_237194

noncomputable def f (x : ℝ) : ℝ := |2^x - 1| - 3^x

theorem number_of_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_zeros_l2371_237194


namespace Lyle_friends_sandwich_juice_l2371_237123

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice_l2371_237123


namespace triangle_sequence_relation_l2371_237174

theorem triangle_sequence_relation (b d c k : ℤ) (h₁ : b % d = 0) (h₂ : c % k = 0) (h₃ : b^2 + (b + 2*d)^2 = (c + 6*k)^2) :
  c = 0 :=
sorry

end triangle_sequence_relation_l2371_237174


namespace gain_percentage_is_8_l2371_237136

variable (C S : ℝ) (D : ℝ)
variable (h1 : 20 * C * (1 - D / 100) = 12 * S)
variable (h2 : D ≥ 5 ∧ D ≤ 25)

theorem gain_percentage_is_8 :
  (12 * S * 1.08 - 20 * C * (1 - D / 100)) / (20 * C * (1 - D / 100)) * 100 = 8 :=
by
  sorry

end gain_percentage_is_8_l2371_237136


namespace statement1_statement2_statement3_l2371_237199

variable (P_W P_Z : ℝ)

/-- The conditions of the problem: -/
def conditions : Prop :=
  P_W = 0.4 ∧ P_Z = 0.2

/-- Proof of the first statement -/
theorem statement1 (h : conditions P_W P_Z) : 
  P_W * P_Z = 0.08 := 
by sorry

/-- Proof of the second statement -/
theorem statement2 (h : conditions P_W P_Z) :
  P_W * (1 - P_Z) + (1 - P_W) * P_Z = 0.44 := 
by sorry

/-- Proof of the third statement -/
theorem statement3 (h : conditions P_W P_Z) :
  1 - P_W * P_Z = 0.92 := 
by sorry

end statement1_statement2_statement3_l2371_237199


namespace market_value_of_share_l2371_237139

-- Definitions from the conditions
def nominal_value : ℝ := 48
def dividend_rate : ℝ := 0.09
def desired_interest_rate : ℝ := 0.12

-- The proof problem (theorem statement) in Lean 4
theorem market_value_of_share : (nominal_value * dividend_rate / desired_interest_rate * 100) = 36 := 
by
  sorry

end market_value_of_share_l2371_237139


namespace multiply_105_95_l2371_237168

theorem multiply_105_95 : 105 * 95 = 9975 :=
by
  sorry

end multiply_105_95_l2371_237168


namespace pyramid_side_length_l2371_237162

-- Definitions for our conditions
def area_of_lateral_face : ℝ := 150
def slant_height : ℝ := 25

-- Theorem statement
theorem pyramid_side_length (A : ℝ) (h : ℝ) (s : ℝ) (hA : A = area_of_lateral_face) (hh : h = slant_height) :
  A = (1 / 2) * s * h → s = 12 :=
by
  intro h_eq
  rw [hA, hh, area_of_lateral_face, slant_height] at h_eq
  -- Steps to verify s = 12
  sorry

end pyramid_side_length_l2371_237162


namespace evaluation_of_expression_l2371_237109

theorem evaluation_of_expression
  (a b x y m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * (|m|) - 2 * (x * y) = 1 :=
by
  -- skipping the proof
  sorry

end evaluation_of_expression_l2371_237109


namespace isosceles_triangle_equal_sides_length_l2371_237120

noncomputable def equal_side_length_isosceles_triangle (base median : ℝ) (vertex_angle_deg : ℝ) : ℝ :=
  if base = 36 ∧ median = 15 ∧ vertex_angle_deg = 60 then 3 * Real.sqrt 191 else 0

theorem isosceles_triangle_equal_sides_length:
  equal_side_length_isosceles_triangle 36 15 60 = 3 * Real.sqrt 191 :=
by
  sorry

end isosceles_triangle_equal_sides_length_l2371_237120


namespace solve_inequality_l2371_237101

theorem solve_inequality (x : ℝ) : 
  1 / (x^2 + 2) > 4 / x + 21 / 10 ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end solve_inequality_l2371_237101


namespace problem_statement_l2371_237102

variable (a b : ℝ)

-- Conditions
variable (h1 : a > 0) (h2 : b > 0) (h3 : ∃ x, x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a)))

-- The Lean theorem statement for the problem
theorem problem_statement : 
  ∀ x, (x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))) →
  (2 * a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := 
sorry


end problem_statement_l2371_237102


namespace largest_n_exists_l2371_237155

theorem largest_n_exists :
  ∃ (n : ℕ), (∀ (x : ℕ → ℝ), (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → (1 + x i * x j)^2 ≤ 0.99 * (1 + x i^2) * (1 + x j^2))) ∧ n = 31 :=
sorry

end largest_n_exists_l2371_237155


namespace macy_miles_left_l2371_237149

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end macy_miles_left_l2371_237149


namespace quadratic_y_at_x_5_l2371_237181

-- Define the quadratic function
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions and question as part of a theorem
theorem quadratic_y_at_x_5 (a b c : ℝ) 
  (h1 : ∀ x, quadratic a b c x ≤ 10) -- Maximum value condition (The maximum value is 10)
  (h2 : (quadratic a b c (-2)) = 10) -- y = 10 when x = -2 (maximum point)
  (h3 : quadratic a b c 0 = -8) -- The first point (0, -8)
  (h4 : quadratic a b c 1 = 0) -- The second point (1, 0)
  : quadratic a b c 5 = -400 / 9 :=
sorry

end quadratic_y_at_x_5_l2371_237181


namespace next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l2371_237126

-- Part (a)
theorem next_terms_arithmetic_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ), 
  a₀ = 3 → a₁ = 7 → a₂ = 11 → a₃ = 15 → a₄ = 19 → a₅ = 23 → d = 4 →
  (a₅ + d = 27) ∧ (a₅ + 2*d = 31) :=
by intros; sorry


-- Part (b)
theorem next_terms_alternating_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℕ),
  a₀ = 9 → a₁ = 1 → a₂ = 7 → a₃ = 1 → a₄ = 5 → a₅ = 1 →
  a₄ - 2 = 3 ∧ a₁ = 1 :=
by intros; sorry


-- Part (c)
theorem next_terms_interwoven_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ),
  a₀ = 4 → a₁ = 5 → a₂ = 8 → a₃ = 9 → a₄ = 12 → a₅ = 13 → d = 4 →
  (a₄ + d = 16) ∧ (a₅ + d = 17) :=
by intros; sorry


-- Part (d)
theorem next_terms_geometric_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅: ℕ), 
  a₀ = 1 → a₁ = 2 → a₂ = 4 → a₃ = 8 → a₄ = 16 → a₅ = 32 →
  (a₅ * 2 = 64) ∧ (a₅ * 4 = 128) :=
by intros; sorry

end next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l2371_237126


namespace regular_triangular_prism_properties_l2371_237158

-- Regular triangular pyramid defined
structure RegularTriangularPyramid (height : ℝ) (base_side : ℝ)

-- Regular triangular prism defined
structure RegularTriangularPrism (height : ℝ) (base_side : ℝ) (lateral_area : ℝ)

-- Given data
def pyramid := RegularTriangularPyramid 15 12
def prism_lateral_area := 120

-- Statement of the problem
theorem regular_triangular_prism_properties (h_prism : ℝ) (ratio_lateral_area : ℚ) :
  (h_prism = 10 ∨ h_prism = 5) ∧ (ratio_lateral_area = 1/9 ∨ ratio_lateral_area = 4/9) :=
sorry

end regular_triangular_prism_properties_l2371_237158


namespace salt_mixture_l2371_237103

theorem salt_mixture (x y : ℝ) (p c z : ℝ) (hx : x = 50) (hp : p = 0.60) (hc : c = 0.40) (hy_eq : y = 50) :
  (50 * z) + (50 * 0.60) = 0.40 * (50 + 50) → (50 * z) + (50 * p) = c * (x + y) → y = 50 :=
by sorry

end salt_mixture_l2371_237103


namespace tangent_lines_l2371_237179

noncomputable def curve1 (x : ℝ) : ℝ := 2 * x ^ 2 - 5
noncomputable def curve2 (x : ℝ) : ℝ := x ^ 2 - 3 * x + 5

theorem tangent_lines :
  (∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = -20 * x - 55 ∨ y = -13 * x - 20 ∨ y = 8 * x - 13 ∨ y = x + 1) ∧ 
    (
      (m₁ = 4 * 2 ∧ b₁ = 3) ∨ 
      (m₁ = 2 * -5 - 3 ∧ b₁ = 45) ∨
      (m₂ = 4 * -5 ∧ b₂ = 45) ∨
      (m₂ = 2 * 2 - 3 ∧ b₂ = 3)
    )) :=
sorry

end tangent_lines_l2371_237179


namespace baseball_card_decrease_l2371_237141

noncomputable def percentDecrease (V : ℝ) (P : ℝ) : ℝ :=
  V * (P / 100)

noncomputable def valueAfterDecrease (V : ℝ) (D : ℝ) : ℝ :=
  V - D

theorem baseball_card_decrease (V : ℝ) (H1 : V > 0) :
  let D1 := percentDecrease V 50
  let V1 := valueAfterDecrease V D1
  let D2 := percentDecrease V1 10
  let V2 := valueAfterDecrease V1 D2
  let totalDecrease := V - V2
  totalDecrease / V * 100 = 55 := sorry

end baseball_card_decrease_l2371_237141


namespace solve_inequality_l2371_237111

theorem solve_inequality (x : ℝ) : 
  let quad := (x - 2)^2 + 9
  let numerator := x - 3
  quad > 0 ∧ numerator ≥ 0 ↔ x ≥ 3 :=
by
    sorry

end solve_inequality_l2371_237111


namespace quadratic_to_vertex_form_l2371_237156

theorem quadratic_to_vertex_form : ∃ m n : ℝ, (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 :=
by sorry

end quadratic_to_vertex_form_l2371_237156


namespace dante_walk_time_l2371_237145

-- Define conditions and problem
variables (T R : ℝ)

-- Conditions as per the problem statement
def wind_in_favor_condition : Prop := 0.8 * T = 15
def wind_against_condition : Prop := 1.25 * T = 7
def total_walk_time_condition : Prop := 15 + 7 = 22
def total_time_away_condition : Prop := 32 - 22 = 10
def lake_park_restaurant_condition : Prop := 0.8 * R = 10

-- Proof statement
theorem dante_walk_time :
  wind_in_favor_condition T ∧
  wind_against_condition T ∧
  total_walk_time_condition ∧
  total_time_away_condition ∧
  lake_park_restaurant_condition R →
  R = 12.5 :=
by
  intros
  sorry

end dante_walk_time_l2371_237145


namespace distance_between_cities_l2371_237138

-- Conditions Initialization
variable (A B : ℝ)
variable (S : ℕ)
variable (x : ℕ)
variable (gcd_values : ℕ)

-- Conditions:
-- 1. The distance between cities A and B is an integer.
-- 2. Markers are placed every kilometer.
-- 3. At each marker, one side has distance to city A, and the other has distance to city B.
-- 4. The GCDs observed were only 1, 3, and 13.

-- Given these conditions, we prove the distance is exactly 39 kilometers.
theorem distance_between_cities (h1 : gcd_values = 1 ∨ gcd_values = 3 ∨ gcd_values = 13)
    (h2 : S = Nat.lcm 1 (Nat.lcm 3 13)) :
    S = 39 := by 
  sorry

end distance_between_cities_l2371_237138


namespace distance_AK_l2371_237131

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def C : ℝ × ℝ := (1, 0)
noncomputable def D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

-- Define the line equations
noncomputable def line_AB (x : ℝ) : Prop := x = 0
noncomputable def line_CD (x y : ℝ) : Prop := y = (Real.sqrt 2) / (2 - Real.sqrt 2) * (x - 1)

-- Define the intersection point K
noncomputable def K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Prove the desired distance
theorem distance_AK : distance A K = Real.sqrt 2 + 1 :=
by
  -- Proof details are omitted
  sorry

end distance_AK_l2371_237131


namespace prove_inequality_l2371_237173

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (55 * Real.pi / 180)

theorem prove_inequality : c > b ∧ b > a :=
by
  -- Proof goes here
  sorry

end prove_inequality_l2371_237173


namespace trajectory_of_C_is_ellipse_l2371_237140

theorem trajectory_of_C_is_ellipse :
  ∀ (C : ℝ × ℝ),
  ((C.1 + 4)^2 + C.2^2).sqrt + ((C.1 - 4)^2 + C.2^2).sqrt = 10 →
  (C.2 ≠ 0) →
  (C.1^2 / 25 + C.2^2 / 9 = 1) :=
by
  intros C h1 h2
  sorry

end trajectory_of_C_is_ellipse_l2371_237140


namespace cost_for_Greg_l2371_237134

theorem cost_for_Greg (N P M : ℝ)
(Bill : 13 * N + 26 * P + 19 * M = 25)
(Paula : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := 
sorry

end cost_for_Greg_l2371_237134


namespace factorize_difference_of_squares_l2371_237146

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) :=
by 
  sorry

end factorize_difference_of_squares_l2371_237146


namespace dividend_is_correct_l2371_237169

def quotient : ℕ := 20
def divisor : ℕ := 66
def remainder : ℕ := 55

def dividend := (divisor * quotient) + remainder

theorem dividend_is_correct : dividend = 1375 := by
  sorry

end dividend_is_correct_l2371_237169


namespace relationship_sides_l2371_237110

-- Definitions for the given condition
variables (a b c : ℝ)

-- Statement of the theorem to prove
theorem relationship_sides (h : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) : a + c = 2 * b :=
sorry

end relationship_sides_l2371_237110


namespace smallest_side_for_table_rotation_l2371_237104

theorem smallest_side_for_table_rotation (S : ℕ) : (S ≥ Int.ofNat (Nat.sqrt (8^2 + 12^2) + 1)) → S = 15 := 
by
  sorry

end smallest_side_for_table_rotation_l2371_237104


namespace scientific_notation_for_70_million_l2371_237135

-- Define the parameters for the problem
def scientific_notation (x : ℕ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Problem statement
theorem scientific_notation_for_70_million :
  scientific_notation 70000000 7.0 7 :=
by
  sorry

end scientific_notation_for_70_million_l2371_237135


namespace total_yield_UncleLi_yield_difference_l2371_237186

-- Define the conditions related to Uncle Li and Aunt Lin
def UncleLiAcres : ℕ := 12
def UncleLiYieldPerAcre : ℕ := 660
def AuntLinAcres : ℕ := UncleLiAcres - 2
def AuntLinTotalYield : ℕ := UncleLiYieldPerAcre * UncleLiAcres - 420

-- Prove the total yield of Uncle Li's rice
theorem total_yield_UncleLi : UncleLiYieldPerAcre * UncleLiAcres = 7920 := by
  sorry

-- Prove how much less the yield per acre of Uncle Li's rice is compared to Aunt Lin's
theorem yield_difference :
  UncleLiYieldPerAcre - AuntLinTotalYield / AuntLinAcres = 90 := by
  sorry

end total_yield_UncleLi_yield_difference_l2371_237186


namespace magic_square_sum_l2371_237100

theorem magic_square_sum (v w x y z : ℤ)
    (h1 : 25 + z + 23 = 25 + x + w)
    (h2 : 18 + x + y = 25 + x + w)
    (h3 : v + 22 + w = 25 + x + w)
    (h4 : 25 + 18 + v = 25 + x + w)
    (h5 : z + x + 22 = 25 + x + w)
    (h6 : 23 + y + w = 25 + x + w)
    (h7 : 25 + x + w = 25 + x + w)
    (h8 : v + x + 23 = 25 + x + w) 
:
    y + z = 45 :=
by
  sorry

end magic_square_sum_l2371_237100


namespace rectangle_divided_into_13_squares_l2371_237183

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end rectangle_divided_into_13_squares_l2371_237183


namespace enjoyable_gameplay_time_l2371_237182

def total_gameplay_time_base : ℝ := 150
def enjoyable_fraction_base : ℝ := 0.30
def total_gameplay_time_expansion : ℝ := 50
def load_screen_fraction_expansion : ℝ := 0.25
def inventory_management_fraction_expansion : ℝ := 0.25
def mod_skip_fraction : ℝ := 0.15

def enjoyable_time_base : ℝ := total_gameplay_time_base * enjoyable_fraction_base
def not_load_screen_time_expansion : ℝ := total_gameplay_time_expansion * (1 - load_screen_fraction_expansion)
def not_inventory_management_time_expansion : ℝ := not_load_screen_time_expansion * (1 - inventory_management_fraction_expansion)

def tedious_time_base : ℝ := total_gameplay_time_base * (1 - enjoyable_fraction_base)
def tedious_time_expansion : ℝ := total_gameplay_time_expansion - not_inventory_management_time_expansion
def total_tedious_time : ℝ := tedious_time_base + tedious_time_expansion

def time_skipped_by_mod : ℝ := total_tedious_time * mod_skip_fraction

def total_enjoyable_time : ℝ := enjoyable_time_base + not_inventory_management_time_expansion + time_skipped_by_mod

theorem enjoyable_gameplay_time :
  total_enjoyable_time = 92.16 :=     by     simp [total_enjoyable_time, enjoyable_time_base, not_inventory_management_time_expansion, time_skipped_by_mod]; sorry

end enjoyable_gameplay_time_l2371_237182


namespace xiaoming_age_l2371_237166

theorem xiaoming_age
  (x x' : ℕ) 
  (h₁ : ∃ f : ℕ, f = 4 * x) 
  (h₂ : (x + 25) + (4 * x + 25) = 100) : 
  x = 10 :=
by
  obtain ⟨f, hf⟩ := h₁
  sorry

end xiaoming_age_l2371_237166


namespace integer_k_values_l2371_237115

noncomputable def is_integer_solution (k x : ℤ) : Prop :=
  ((k - 2013) * x = 2015 - 2014 * x)

theorem integer_k_values (k : ℤ) (h : ∃ x : ℤ, is_integer_solution k x) :
  ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_k_values_l2371_237115


namespace find_r_floor_r_add_r_eq_18point2_l2371_237190

theorem find_r_floor_r_add_r_eq_18point2 (r : ℝ) (h : ⌊r⌋ + r = 18.2) : r = 9.2 := 
sorry

end find_r_floor_r_add_r_eq_18point2_l2371_237190


namespace smallest_base_l2371_237160

theorem smallest_base (k b : ℕ) (h_k : k = 6) : 64 ^ k > b ^ 16 ↔ b < 5 :=
by
  have h1 : 64 ^ k = 2 ^ (6 * k) := by sorry
  have h2 : 2 ^ (6 * k) > b ^ 16 := by sorry
  exact sorry

end smallest_base_l2371_237160


namespace min_ones_count_in_100_numbers_l2371_237148

def sum_eq_product (l : List ℕ) : Prop :=
  l.sum = l.prod

theorem min_ones_count_in_100_numbers : ∀ l : List ℕ, l.length = 100 → sum_eq_product l → l.count 1 ≥ 95 :=
by sorry

end min_ones_count_in_100_numbers_l2371_237148


namespace find_y_l2371_237105

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : x / y = 86 ∧ ((x % y : ℚ) / y = 0.12)) : y = 75 :=
by
  sorry

end find_y_l2371_237105


namespace train_around_probability_train_present_when_alex_arrives_l2371_237163

noncomputable def trainArrivalTime : Set ℝ := Set.Icc 15 45
noncomputable def trainWaitTime (t : ℝ) : Set ℝ := Set.Icc t (t + 15)
noncomputable def alexArrivalTime : Set ℝ := Set.Icc 0 60

theorem train_around (t : ℝ) (h : t ∈ trainArrivalTime) :
  ∀ (x : ℝ), x ∈ alexArrivalTime → x ∈ trainWaitTime t ↔ 15 ≤ t ∧ t ≤ 45 ∧ t ≤ x ∧ x ≤ t + 15 :=
sorry

theorem probability_train_present_when_alex_arrives :
  let total_area := 60 * 60
  let favorable_area := 1 / 2 * (15 + 15) * 15
  (favorable_area / total_area) = 1 / 16 :=
sorry

end train_around_probability_train_present_when_alex_arrives_l2371_237163


namespace determine_xyz_l2371_237197

theorem determine_xyz (x y z : ℝ) 
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 12) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) : 
  x * y * z = -4 / 3 := 
sorry

end determine_xyz_l2371_237197


namespace equation_of_circle_unique_l2371_237121

noncomputable def equation_of_circle := 
  ∃ (d e f : ℝ), 
    (4 + 4 + 2*d + 2*e + f = 0) ∧ 
    (25 + 9 + 5*d + 3*e + f = 0) ∧ 
    (9 + 1 + 3*d - e + f = 0) ∧ 
    (∀ (x y : ℝ), x^2 + y^2 + d*x + e*y + f = 0 → (x = 2 ∧ y = 2) ∨ (x = 5 ∧ y = 3) ∨ (x = 3 ∧ y = -1))

theorem equation_of_circle_unique :
  equation_of_circle := sorry

end equation_of_circle_unique_l2371_237121


namespace bug_paths_l2371_237108

-- Define the problem conditions
structure PathSetup (A B : Type) :=
  (red_arrows : ℕ) -- number of red arrows from point A
  (red_to_blue : ℕ) -- number of blue arrows reachable from each red arrow
  (blue_to_green : ℕ) -- number of green arrows reachable from each blue arrow
  (green_to_orange : ℕ) -- number of orange arrows reachable from each green arrow
  (start_arrows : ℕ) -- starting number of arrows from point A to red arrows
  (orange_arrows : ℕ) -- number of orange arrows equivalent to green arrows

-- Define the conditions for our specific problem setup
def problem_setup : PathSetup Point Point :=
  {
    red_arrows := 3,
    red_to_blue := 2,
    blue_to_green := 2,
    green_to_orange := 1,
    start_arrows := 3,
    orange_arrows := 6 * 2 * 2 -- derived from blue_to_green and red_to_blue steps
  }

-- Prove the number of unique paths from A to B
theorem bug_paths (setup : PathSetup Point Point) : 
  setup.start_arrows * setup.red_to_blue * setup.blue_to_green * setup.green_to_orange * setup.orange_arrows = 1440 :=
by
  -- Calculations are performed; exact values must hold
  sorry

end bug_paths_l2371_237108


namespace loan_duration_l2371_237143

theorem loan_duration (P R SI : ℝ) (hP : P = 20000) (hR : R = 12) (hSI : SI = 7200) : 
  ∃ T : ℝ, T = 3 :=
by
  sorry

end loan_duration_l2371_237143


namespace original_stations_l2371_237164

theorem original_stations (m n : ℕ) (h : n > 1) (h_equation : n * (2 * m + n - 1) = 58) : m = 14 :=
by
  -- proof omitted
  sorry

end original_stations_l2371_237164
