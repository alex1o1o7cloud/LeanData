import Mathlib

namespace NUMINAMATH_GPT_abs_x_plus_2_l491_49132

theorem abs_x_plus_2 (x : ℤ) (h : x = -3) : |x + 2| = 1 :=
by sorry

end NUMINAMATH_GPT_abs_x_plus_2_l491_49132


namespace NUMINAMATH_GPT_betty_needs_five_boxes_l491_49119

def betty_oranges (total_oranges first_box second_box max_per_box : ℕ) : ℕ :=
  let remaining_oranges := total_oranges - (first_box + second_box)
  let full_boxes := remaining_oranges / max_per_box
  let extra_box := if remaining_oranges % max_per_box == 0 then 0 else 1
  full_boxes + 2 + extra_box

theorem betty_needs_five_boxes :
  betty_oranges 120 30 25 30 = 5 := 
by
  sorry

end NUMINAMATH_GPT_betty_needs_five_boxes_l491_49119


namespace NUMINAMATH_GPT_range_of_a_l491_49199

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, (2 * (x : ℝ) - 7 < 0) ∧ ((x : ℝ) - a > 0) ↔ (x = 3)) →
  (2 ≤ a ∧ a < 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l491_49199


namespace NUMINAMATH_GPT_range_of_m_l491_49127

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : 
  (x + y ≥ m) → m ≤ 18 :=
sorry

end NUMINAMATH_GPT_range_of_m_l491_49127


namespace NUMINAMATH_GPT_investment_at_6_percent_l491_49100

theorem investment_at_6_percent
  (x y : ℝ) 
  (total_investment : x + y = 15000)
  (total_interest : 0.06 * x + 0.075 * y = 1023) :
  x = 6800 :=
sorry

end NUMINAMATH_GPT_investment_at_6_percent_l491_49100


namespace NUMINAMATH_GPT_weather_forecast_probability_l491_49183

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem weather_forecast_probability :
  binomial_probability 3 2 0.8 = 0.384 :=
by
  sorry

end NUMINAMATH_GPT_weather_forecast_probability_l491_49183


namespace NUMINAMATH_GPT_christen_potatoes_peeled_l491_49108

-- Define the initial conditions and setup
def initial_potatoes := 50
def homer_rate := 4
def christen_rate := 6
def time_homer_alone := 5
def combined_rate := homer_rate + christen_rate

-- Calculate the number of potatoes peeled by Homer alone in the first 5 minutes
def potatoes_peeled_by_homer_alone := time_homer_alone * homer_rate

-- Calculate the remaining potatoes after Homer peeled alone
def remaining_potatoes := initial_potatoes - potatoes_peeled_by_homer_alone

-- Calculate the time taken for Homer and Christen to peel the remaining potatoes together
def time_to_finish_together := remaining_potatoes / combined_rate

-- Calculate the number of potatoes peeled by Christen during the shared work period
def potatoes_peeled_by_christen := christen_rate * time_to_finish_together

-- The final theorem we need to prove
theorem christen_potatoes_peeled : potatoes_peeled_by_christen = 18 := by
  sorry

end NUMINAMATH_GPT_christen_potatoes_peeled_l491_49108


namespace NUMINAMATH_GPT_not_54_after_one_hour_l491_49155

theorem not_54_after_one_hour (n : ℕ) (initial_number : ℕ) (initial_factors : ℕ × ℕ)
  (h₀ : initial_number = 12)
  (h₁ : initial_factors = (2, 1)) :
  (∀ k : ℕ, k < 60 →
    ∀ current_factors : ℕ × ℕ,
    current_factors = (initial_factors.1 + k, initial_factors.2 + k) ∨
    current_factors = (initial_factors.1 - k, initial_factors.2 - k) →
    initial_number * (2 ^ (initial_factors.1 + k)) * (3 ^ (initial_factors.2 + k)) ≠ 54) :=
by
  sorry

end NUMINAMATH_GPT_not_54_after_one_hour_l491_49155


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l491_49147

theorem no_real_roots_of_quadratic (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b ≠ 0) ↔ ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l491_49147


namespace NUMINAMATH_GPT_nests_count_l491_49158

theorem nests_count :
  ∃ (N : ℕ), (6 = N + 3) ∧ (N = 3) :=
by
  sorry

end NUMINAMATH_GPT_nests_count_l491_49158


namespace NUMINAMATH_GPT_bryan_initial_pushups_l491_49139

def bryan_pushups (x : ℕ) : Prop :=
  let totalPushups := x + x + (x - 5)
  totalPushups = 40

theorem bryan_initial_pushups (x : ℕ) (hx : bryan_pushups x) : x = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_bryan_initial_pushups_l491_49139


namespace NUMINAMATH_GPT_watched_videos_correct_l491_49174

-- Conditions
def num_suggestions_per_time : ℕ := 15
def times : ℕ := 5
def chosen_position : ℕ := 5

-- Question
def total_videos_watched : ℕ := num_suggestions_per_time * times - (num_suggestions_per_time - chosen_position)

-- Proof
theorem watched_videos_correct : total_videos_watched = 65 := by
  sorry

end NUMINAMATH_GPT_watched_videos_correct_l491_49174


namespace NUMINAMATH_GPT_harvesting_days_l491_49123

theorem harvesting_days :
  (∀ (harvesters : ℕ) (days : ℕ) (mu : ℕ), 2 * 3 * (75 : ℕ) = 450) →
  (7 * 4 * (75 : ℕ) = 2100) :=
by
  sorry

end NUMINAMATH_GPT_harvesting_days_l491_49123


namespace NUMINAMATH_GPT_trail_mix_total_weight_l491_49144

def peanuts : ℝ := 0.16666666666666666
def chocolate_chips : ℝ := 0.16666666666666666
def raisins : ℝ := 0.08333333333333333
def trail_mix_weight : ℝ := 0.41666666666666663

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = trail_mix_weight :=
sorry

end NUMINAMATH_GPT_trail_mix_total_weight_l491_49144


namespace NUMINAMATH_GPT_anika_age_l491_49136

/-- Given:
 1. Anika is 10 years younger than Clara.
 2. Clara is 5 years older than Ben.
 3. Ben is 20 years old.
 Prove:
 Anika's age is 15 years.
 -/
theorem anika_age (Clara Anika Ben : ℕ) 
  (h1 : Anika = Clara - 10) 
  (h2 : Clara = Ben + 5) 
  (h3 : Ben = 20) : Anika = 15 := 
by
  sorry

end NUMINAMATH_GPT_anika_age_l491_49136


namespace NUMINAMATH_GPT_greatest_two_digit_with_product_12_l491_49194

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end NUMINAMATH_GPT_greatest_two_digit_with_product_12_l491_49194


namespace NUMINAMATH_GPT_coefficient_a9_of_polynomial_l491_49175

theorem coefficient_a9_of_polynomial (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a_0 + 
    a_1 * (x + 1) + 
    a_2 * (x + 1)^2 + 
    a_3 * (x + 1)^3 + 
    a_4 * (x + 1)^4 + 
    a_5 * (x + 1)^5 + 
    a_6 * (x + 1)^6 + 
    a_7 * (x + 1)^7 + 
    a_8 * (x + 1)^8 + 
    a_9 * (x + 1)^9 + 
    a_10 * (x + 1)^10) 
  → a_9 = -10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_coefficient_a9_of_polynomial_l491_49175


namespace NUMINAMATH_GPT_union_of_sets_l491_49193

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2, 6}) (hB : B = {2, 3, 6}) :
  A ∪ B = {1, 2, 3, 6} :=
by
  rw [hA, hB]
  ext x
  simp [Set.union]
  sorry

end NUMINAMATH_GPT_union_of_sets_l491_49193


namespace NUMINAMATH_GPT_find_t_l491_49164

open Real

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (t : ℝ) :
  let m := (t + 1, 1)
  let n := (t + 2, 2)
  dot_product (vector_add m n) (vector_sub m n) = 0 → 
  t = -3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_t_l491_49164


namespace NUMINAMATH_GPT_storm_deposit_l491_49159

theorem storm_deposit (C : ℝ) (original_amount after_storm_rate before_storm_rate : ℝ) (after_storm full_capacity : ℝ) :
  before_storm_rate = 0.40 →
  after_storm_rate = 0.60 →
  original_amount = 220 * 10^9 →
  before_storm_rate * C = original_amount →
  C = full_capacity →
  after_storm = after_storm_rate * full_capacity →
  after_storm - original_amount = 110 * 10^9 :=
by
  sorry

end NUMINAMATH_GPT_storm_deposit_l491_49159


namespace NUMINAMATH_GPT_number_of_triangles_l491_49160

theorem number_of_triangles (points : List ℝ) (h₀ : points.length = 12)
  (h₁ : ∀ p ∈ points, p ≠ A ∧ p ≠ B ∧ p ≠ C ∧ p ≠ D): 
  (∃ triangles : ℕ, triangles = 216) :=
  sorry

end NUMINAMATH_GPT_number_of_triangles_l491_49160


namespace NUMINAMATH_GPT_extreme_value_range_of_a_l491_49141

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) * (1 - a * x)

theorem extreme_value_range_of_a (a : ℝ) :
  a ∈ Set.Ioo (2 / 3 : ℝ) 2 ↔
    ∃ c ∈ Set.Ioo 0 1, ∀ x : ℝ, f a c = f a x :=
by
  sorry

end NUMINAMATH_GPT_extreme_value_range_of_a_l491_49141


namespace NUMINAMATH_GPT_Brittany_older_by_3_years_l491_49178

-- Define the necessary parameters as assumptions
variable (Rebecca_age : ℕ) (Brittany_return_age : ℕ) (vacation_years : ℕ)

-- Initial conditions
axiom h1 : Rebecca_age = 25
axiom h2 : Brittany_return_age = 32
axiom h3 : vacation_years = 4

-- Definition to capture Brittany's age before vacation
def Brittany_age_before_vacation (return_age vacation_period : ℕ) : ℕ := return_age - vacation_period

-- Theorem stating that Brittany is 3 years older than Rebecca
theorem Brittany_older_by_3_years :
  Brittany_age_before_vacation Brittany_return_age vacation_years - Rebecca_age = 3 :=
by
  sorry

end NUMINAMATH_GPT_Brittany_older_by_3_years_l491_49178


namespace NUMINAMATH_GPT_problem1_problem2_l491_49110

theorem problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x + y ≥ 6 :=
sorry

theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x * y ≥ 9 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l491_49110


namespace NUMINAMATH_GPT_dot_product_eq_neg29_l491_49188

def v := (3, -2)
def w := (-5, 7)

theorem dot_product_eq_neg29 : (v.1 * w.1 + v.2 * w.2) = -29 := 
by 
  -- this is where the detailed proof will occur
  sorry

end NUMINAMATH_GPT_dot_product_eq_neg29_l491_49188


namespace NUMINAMATH_GPT_sum_of_first_100_terms_l491_49179

theorem sum_of_first_100_terms (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n+2) = a n + 1) : 
  (Finset.sum (Finset.range 100) a) = 2550 :=
sorry

end NUMINAMATH_GPT_sum_of_first_100_terms_l491_49179


namespace NUMINAMATH_GPT_triangle_B_eq_2A_range_of_a_l491_49126

theorem triangle_B_eq_2A (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = c) : B = 2 * A := 
sorry

theorem range_of_a (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = 2) (h6 : 0 < (π - A - B)) (h7 : (π - A - B) < π/2) : 1 < a ∧ a < 2 := 
sorry

end NUMINAMATH_GPT_triangle_B_eq_2A_range_of_a_l491_49126


namespace NUMINAMATH_GPT_alpha_is_30_or_60_l491_49170

theorem alpha_is_30_or_60
  (α : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2) -- α is acute angle
  (a : ℝ × ℝ := (3 / 4, Real.sin α))
  (b : ℝ × ℝ := (Real.cos α, 1 / Real.sqrt 3))
  (h2 : a.1 * b.2 = a.2 * b.1)  -- a ∥ b
  : α = Real.pi / 6 ∨ α = Real.pi / 3 := 
sorry

end NUMINAMATH_GPT_alpha_is_30_or_60_l491_49170


namespace NUMINAMATH_GPT_sum_of_squares_positive_l491_49148

theorem sum_of_squares_positive (x_1 x_2 k : ℝ) (h : x_1 ≠ x_2) 
  (hx1 : x_1^2 + 2*x_1 - k = 0) (hx2 : x_2^2 + 2*x_2 - k = 0) :
  x_1^2 + x_2^2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_positive_l491_49148


namespace NUMINAMATH_GPT_hula_hoop_ratio_l491_49124

variable (Nancy Casey Morgan : ℕ)
variable (hula_hoop_time_Nancy : Nancy = 10)
variable (hula_hoop_time_Casey : Casey = Nancy - 3)
variable (hula_hoop_time_Morgan : Morgan = 21)

theorem hula_hoop_ratio (hula_hoop_time_Nancy : Nancy = 10) (hula_hoop_time_Casey : Casey = Nancy - 3) (hula_hoop_time_Morgan : Morgan = 21) :
  Morgan / Casey = 3 := by
  sorry

end NUMINAMATH_GPT_hula_hoop_ratio_l491_49124


namespace NUMINAMATH_GPT_cricket_matches_total_l491_49180

theorem cricket_matches_total 
  (N : ℕ)
  (avg_total : ℕ → ℕ)
  (avg_first_8 : ℕ)
  (avg_last_4 : ℕ) 
  (h1 : avg_total N = 48)
  (h2 : avg_first_8 = 40)
  (h3 : avg_last_4 = 64) 
  (h_sum : (avg_first_8 * 8 + avg_last_4 * 4 = avg_total N * N)) :
  N = 12 := 
  sorry

end NUMINAMATH_GPT_cricket_matches_total_l491_49180


namespace NUMINAMATH_GPT_second_quadrant_condition_l491_49120

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_in_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ -270 < α ∧ α < -180

theorem second_quadrant_condition (α : ℝ) : 
  (is_obtuse α → is_in_second_quadrant α) ∧ ¬(is_in_second_quadrant α → is_obtuse α) := 
by
  sorry

end NUMINAMATH_GPT_second_quadrant_condition_l491_49120


namespace NUMINAMATH_GPT_dobarulho_problem_l491_49149

def is_divisible_by (x d : ℕ) : Prop := d ∣ x

def valid_quadruple (A B C D : ℕ) : Prop :=
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (A ≤ 8) ∧ (D > 1) ∧
  is_divisible_by (100 * A + 10 * B + C) D ∧
  is_divisible_by (100 * B + 10 * C + A) D ∧
  is_divisible_by (100 * C + 10 * A + B) D ∧
  is_divisible_by (100 * (A + 1) + 10 * C + B) D ∧
  is_divisible_by (100 * C + 10 * B + (A + 1)) D ∧
  is_divisible_by (100 * B + 10 * (A + 1) + C) D 

theorem dobarulho_problem :
  ∀ (A B C D : ℕ), valid_quadruple A B C D → 
  (A = 3 ∧ B = 7 ∧ C = 0 ∧ D = 37) ∨ 
  (A = 4 ∧ B = 8 ∧ C = 1 ∧ D = 37) ∨
  (A = 5 ∧ B = 9 ∧ C = 2 ∧ D = 37) :=
by sorry

end NUMINAMATH_GPT_dobarulho_problem_l491_49149


namespace NUMINAMATH_GPT_cost_of_pen_is_five_l491_49131

-- Define the given conditions
def pencils_per_box := 80
def num_boxes := 15
def total_pencils := num_boxes * pencils_per_box
def cost_per_pencil := 4
def total_cost_of_stationery := 18300
def additional_pens := 300
def num_pens := 2 * total_pencils + additional_pens

-- Calculate total cost of pencils
def total_cost_of_pencils := total_pencils * cost_per_pencil

-- Calculate total cost of pens
def total_cost_of_pens := total_cost_of_stationery - total_cost_of_pencils

-- The conjecture to prove
theorem cost_of_pen_is_five :
  (total_cost_of_pens / num_pens) = 5 :=
sorry

end NUMINAMATH_GPT_cost_of_pen_is_five_l491_49131


namespace NUMINAMATH_GPT_maria_total_earnings_l491_49154

-- Definitions of the conditions
def day1_tulips := 30
def day1_roses := 20
def day2_tulips := 2 * day1_tulips
def day2_roses := 2 * day1_roses
def day3_tulips := day2_tulips / 10
def day3_roses := 16
def tulip_price := 2
def rose_price := 3

-- Definition of the total earnings calculation
noncomputable def total_earnings : ℤ :=
  let total_tulips := day1_tulips + day2_tulips + day3_tulips
  let total_roses := day1_roses + day2_roses + day3_roses
  (total_tulips * tulip_price) + (total_roses * rose_price)

-- The proof statement
theorem maria_total_earnings : total_earnings = 420 := by
  sorry

end NUMINAMATH_GPT_maria_total_earnings_l491_49154


namespace NUMINAMATH_GPT_correct_number_is_650_l491_49125

theorem correct_number_is_650 
  (n : ℕ) 
  (h : n - 152 = 346): 
  n + 152 = 650 :=
by
  sorry

end NUMINAMATH_GPT_correct_number_is_650_l491_49125


namespace NUMINAMATH_GPT_triangle_third_side_lengths_l491_49101

theorem triangle_third_side_lengths : 
  ∃ (x : ℕ), (3 < x ∧ x < 11) ∧ (x ≠ 3) ∧ (x ≠ 11) ∧ 
    ((x = 4) ∨ (x = 5) ∨ (x = 6) ∨ (x = 7) ∨ (x = 8) ∨ (x = 9) ∨ (x = 10)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_lengths_l491_49101


namespace NUMINAMATH_GPT_bus_car_ratio_l491_49137

variable (R C Y : ℝ)

noncomputable def ratio_of_bus_to_car (R C Y : ℝ) : ℝ :=
  R / C

theorem bus_car_ratio 
  (h1 : R = 48) 
  (h2 : Y = 3.5 * C) 
  (h3 : Y = R - 6) : 
  ratio_of_bus_to_car R C Y = 4 :=
by sorry

end NUMINAMATH_GPT_bus_car_ratio_l491_49137


namespace NUMINAMATH_GPT_inclination_angle_of_line_l491_49163

-- Definitions drawn from the condition.
def line_equation (x y : ℝ) := x - y + 1 = 0

-- The statement of the theorem (equivalent proof problem).
theorem inclination_angle_of_line : ∀ x y : ℝ, line_equation x y → θ = π / 4 :=
sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l491_49163


namespace NUMINAMATH_GPT_ab_leq_one_fraction_inequality_l491_49116

-- Part 1: Prove that ab ≤ 1
theorem ab_leq_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) : a * b ≤ 1 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

-- Part 2: Prove that (1/a^3 - 1/b^3) > 3 * (1/a - 1/b) given b > a
theorem fraction_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) (h4 : b > a) :
  1/(a^3) - 1/(b^3) > 3 * (1/a - 1/b) :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end NUMINAMATH_GPT_ab_leq_one_fraction_inequality_l491_49116


namespace NUMINAMATH_GPT_find_min_sum_of_squares_l491_49187

open Real

theorem find_min_sum_of_squares
  (x1 x2 x3 : ℝ)
  (h1 : 0 < x1)
  (h2 : 0 < x2)
  (h3 : 0 < x3)
  (h4 : 2 * x1 + 4 * x2 + 6 * x3 = 120) :
  x1^2 + x2^2 + x3^2 >= 350 :=
sorry

end NUMINAMATH_GPT_find_min_sum_of_squares_l491_49187


namespace NUMINAMATH_GPT_fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l491_49185

theorem fraction_inequalities (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1 / 2 ≤ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ∧ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ≤ 1 :=
sorry

theorem fraction_inequality_equality_right (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1 - a) * (1 - b) = 0 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) :=
sorry

theorem fraction_inequality_equality_left (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  a = b ↔ a = 1 / 2 ∧ b = 1 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l491_49185


namespace NUMINAMATH_GPT_total_keys_needed_l491_49190

-- Definitions based on given conditions
def num_complexes : ℕ := 2
def num_apartments_per_complex : ℕ := 12
def keys_per_lock : ℕ := 3
def num_locks_per_apartment : ℕ := 1

-- Theorem stating the required number of keys
theorem total_keys_needed : 
  (num_complexes * num_apartments_per_complex * keys_per_lock = 72) :=
by
  sorry

end NUMINAMATH_GPT_total_keys_needed_l491_49190


namespace NUMINAMATH_GPT_parallel_lines_necessary_not_sufficient_l491_49117

theorem parallel_lines_necessary_not_sufficient {a : ℝ} 
  (h1 : ∀ x y : ℝ, a * x + (a + 2) * y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + a * y + 2 = 0) 
  (h3 : ∀ x y : ℝ, a * (1 * y + 2) = 1 * (a * y + 2)) : 
  (a = -1) -> (a = 2 ∨ a = -1 ∧ ¬(∀ b, a = b → a = -1)) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_parallel_lines_necessary_not_sufficient_l491_49117


namespace NUMINAMATH_GPT_total_workers_l491_49172

theorem total_workers (h_beavers : ℕ := 318) (h_spiders : ℕ := 544) :
  h_beavers + h_spiders = 862 :=
by
  sorry

end NUMINAMATH_GPT_total_workers_l491_49172


namespace NUMINAMATH_GPT_sum_of_three_numbers_l491_49107

theorem sum_of_three_numbers (a b c : ℝ) (h1 : (a + b + c) / 3 = a - 15) (h2 : (a + b + c) / 3 = c + 10) (h3 : b = 10) :
  a + b + c = 45 :=
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l491_49107


namespace NUMINAMATH_GPT_magnitude_of_z_8_l491_49169

def z : Complex := 2 + 3 * Complex.I

theorem magnitude_of_z_8 : Complex.abs (z ^ 8) = 28561 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_8_l491_49169


namespace NUMINAMATH_GPT_minimum_red_chips_l491_49143

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ w / 4) (h2 : b ≤ r / 6) (h3 : w + b ≥ 75) : r ≥ 90 :=
sorry

end NUMINAMATH_GPT_minimum_red_chips_l491_49143


namespace NUMINAMATH_GPT_missing_digit_in_138_x_6_divisible_by_9_l491_49177

theorem missing_digit_in_138_x_6_divisible_by_9 :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ (1 + 3 + 8 + x + 6) % 9 = 0 ∧ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_missing_digit_in_138_x_6_divisible_by_9_l491_49177


namespace NUMINAMATH_GPT_inequality_solution_l491_49171

theorem inequality_solution (x : ℝ) :
  (x - 2 > 1) ∧ (-2 * x ≤ 4) ↔ (x > 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l491_49171


namespace NUMINAMATH_GPT_quadratic_coefficients_l491_49189

theorem quadratic_coefficients :
  ∃ a b c : ℤ, a = 4 ∧ b = 0 ∧ c = -3 ∧ 4 * x^2 = 3 := sorry

end NUMINAMATH_GPT_quadratic_coefficients_l491_49189


namespace NUMINAMATH_GPT_fraction_of_tea_in_final_cup2_is_5_over_8_l491_49140

-- Defining the initial conditions and the transfers
structure CupContents where
  tea : ℚ
  milk : ℚ

def initialCup1 : CupContents := { tea := 6, milk := 0 }
def initialCup2 : CupContents := { tea := 0, milk := 3 }

def transferOneThird (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let teaTransferred := (1 / 3) * cup1.tea
  ( { cup1 with tea := cup1.tea - teaTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk } )

def transferOneFourth (cup2 : CupContents) (cup1 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup2.tea + cup2.milk
  let amountTransferred := (1 / 4) * mixedTotal
  let teaTransferred := amountTransferred * (cup2.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup2.milk / mixedTotal)
  ( { tea := cup1.tea + teaTransferred, milk := cup1.milk + milkTransferred },
    { tea := cup2.tea - teaTransferred, milk := cup2.milk - milkTransferred } )

def transferOneHalf (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup1.tea + cup1.milk
  let amountTransferred := (1 / 2) * mixedTotal
  let teaTransferred := amountTransferred * (cup1.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup1.milk / mixedTotal)
  ( { tea := cup1.tea - teaTransferred, milk := cup1.milk - milkTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk + milkTransferred } )

def finalContents (cup1 cup2 : CupContents) : CupContents × CupContents :=
  let (cup1Transferred, cup2Transferred) := transferOneThird cup1 cup2
  let (cup1Mixed, cup2Mixed) := transferOneFourth cup2Transferred cup1Transferred
  transferOneHalf cup1Mixed cup2Mixed

-- Statement to be proved
theorem fraction_of_tea_in_final_cup2_is_5_over_8 :
  ((finalContents initialCup1 initialCup2).snd.tea / ((finalContents initialCup1 initialCup2).snd.tea + (finalContents initialCup1 initialCup2).snd.milk) = 5 / 8) :=
sorry

end NUMINAMATH_GPT_fraction_of_tea_in_final_cup2_is_5_over_8_l491_49140


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l491_49103

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n, a n = 2 * n - 1 := 
by
  intros h1 h2 n
  sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l491_49103


namespace NUMINAMATH_GPT_least_number_added_to_divide_l491_49150

-- Definitions of conditions
def lcm_three_five_seven_eight : ℕ := Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 8
def remainder_28523_lcm := 28523 % lcm_three_five_seven_eight

-- Lean statement to prove the correct answer
theorem least_number_added_to_divide (n : ℕ) :
  n = lcm_three_five_seven_eight - remainder_28523_lcm :=
sorry

end NUMINAMATH_GPT_least_number_added_to_divide_l491_49150


namespace NUMINAMATH_GPT_parabola_equation_origin_l491_49173

theorem parabola_equation_origin (x0 : ℝ) :
  ∃ (p : ℝ), (p > 0) ∧ (x0^2 = 2 * p * 2) ∧ (p = 2) ∧ (x0^2 = 4 * 2) := 
by 
  sorry

end NUMINAMATH_GPT_parabola_equation_origin_l491_49173


namespace NUMINAMATH_GPT_sequence_no_limit_l491_49162

noncomputable def sequence_limit (x : ℕ → ℝ) (a : ℝ) : Prop :=
    ∀ ε > 0, ∃ N, ∀ n > N, abs (x n - a) < ε

theorem sequence_no_limit (x : ℕ → ℝ) (a : ℝ) (ε : ℝ) (k : ℕ) :
    (ε > 0) ∧ (∀ n, n > k → abs (x n - a) ≥ ε) → ¬ sequence_limit x a :=
by
  sorry

end NUMINAMATH_GPT_sequence_no_limit_l491_49162


namespace NUMINAMATH_GPT_part1_part2_l491_49104

def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2 * a|

-- Define the first part of the problem
theorem part1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
sorry

-- Define the second part of the problem
theorem part2 (a x : ℝ) (h1 : a ≥ 1) : f x a ≥ 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l491_49104


namespace NUMINAMATH_GPT_relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l491_49166

-- Let a, b, and c be the sides of a right triangle with c as the hypotenuse.
variables (a b c : ℝ) (n : ℕ)

-- Assume the triangle is a right triangle
-- and assume n is a positive integer.
axiom right_triangle : a^2 + b^2 = c^2
axiom positive_integer : n > 0 

-- The relationships we need to prove:
theorem relationship_a_plus_b_greater_c : n = 1 → a + b > c := sorry
theorem relationship_a_squared_plus_b_squared_equals_c_squared : n = 2 → a^2 + b^2 = c^2 := sorry
theorem relationship_a_n_plus_b_n_less_than_c_n : n ≥ 3 → a^n + b^n < c^n := sorry

end NUMINAMATH_GPT_relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l491_49166


namespace NUMINAMATH_GPT_factor_polynomial_l491_49195

theorem factor_polynomial :
  (x^2 + 5 * x + 4) * (x^2 + 11 * x + 30) + (x^2 + 8 * x - 10) =
  (x^2 + 8 * x + 7) * (x^2 + 8 * x + 19) := by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l491_49195


namespace NUMINAMATH_GPT_find_mnp_l491_49102

noncomputable def equation_rewrite (a b x y : ℝ) (m n p : ℕ): Prop :=
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) ∧
  (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5

theorem find_mnp (a b x y : ℝ): 
  equation_rewrite a b x y 2 1 4 ∧ (2 * 1 * 4 = 8) :=
by 
  sorry

end NUMINAMATH_GPT_find_mnp_l491_49102


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l491_49142

theorem geometric_sequence_common_ratio
  (q a_1 : ℝ)
  (h1: a_1 * q = 1)
  (h2: a_1 + a_1 * q^2 = -2) :
  q = -1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l491_49142


namespace NUMINAMATH_GPT_Amanda_hiking_trip_l491_49111

-- Define the conditions
variable (x : ℝ) -- the total distance of Amanda's hiking trip
variable (forest_path : ℝ) (plain_path : ℝ)
variable (stream_path : ℝ) (mountain_path : ℝ)

-- Given conditions
axiom h1 : stream_path = (1/4) * x
axiom h2 : forest_path = 25
axiom h3 : mountain_path = (1/6) * x
axiom h4 : plain_path = 2 * forest_path
axiom h5 : stream_path + forest_path + mountain_path + plain_path = x

-- Proposition to prove
theorem Amanda_hiking_trip : x = 900 / 7 :=
by
  sorry

end NUMINAMATH_GPT_Amanda_hiking_trip_l491_49111


namespace NUMINAMATH_GPT_find_m_l491_49197

-- Definitions for vectors and dot products
structure Vector :=
  (i : ℝ)
  (j : ℝ)

def dot_product (a b : Vector) : ℝ :=
  a.i * b.i + a.j * b.j

-- Given conditions
def i : Vector := ⟨1, 0⟩
def j : Vector := ⟨0, 1⟩

def a : Vector := ⟨2, 3⟩
def b (m : ℝ) : Vector := ⟨1, -m⟩

-- The main goal
theorem find_m (m : ℝ) (h: dot_product a (b m) = 1) : m = 1 / 3 :=
by {
  -- Calculation reaches the same \(m = 1/3\)
  sorry
}

end NUMINAMATH_GPT_find_m_l491_49197


namespace NUMINAMATH_GPT_geom_seq_expression_l491_49184

theorem geom_seq_expression (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 10) (h2 : a 2 + a 4 = 5) :
  ∀ n, a n = 2 ^ (4 - n) :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_geom_seq_expression_l491_49184


namespace NUMINAMATH_GPT_value_of_expression_l491_49182

theorem value_of_expression :
  let x := 1
  let y := -1
  let z := 0
  2 * x + 3 * y + 4 * z = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l491_49182


namespace NUMINAMATH_GPT_sum_mod_9_l491_49165

theorem sum_mod_9 (x y z : ℕ) (h1 : x < 9) (h2 : y < 9) (h3 : z < 9) 
  (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : (x * y * z) % 9 = 1) (h8 : (7 * z) % 9 = 4) (h9 : (8 * y) % 9 = (5 + y) % 9) :
  (x + y + z) % 9 = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_mod_9_l491_49165


namespace NUMINAMATH_GPT_solve_x_l491_49138

theorem solve_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.84) : x = 72 := 
by
  sorry

end NUMINAMATH_GPT_solve_x_l491_49138


namespace NUMINAMATH_GPT_range_of_t_l491_49129

theorem range_of_t (t : ℝ) (h : ∃ x : ℝ, x ∈ Set.Iic t ∧ (x^2 - 4*x + t ≤ 0)) : 0 ≤ t ∧ t ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_t_l491_49129


namespace NUMINAMATH_GPT_father_l491_49196

variable {son_age : ℕ} -- Son's present age
variable {father_age : ℕ} -- Father's present age

-- Conditions
def father_is_four_times_son (son_age father_age : ℕ) : Prop := father_age = 4 * son_age
def sum_of_ages_ten_years_ago (son_age father_age : ℕ) : Prop := (son_age - 10) + (father_age - 10) = 60

-- Theorem statement
theorem father's_present_age 
  (son_age father_age : ℕ)
  (h1 : father_is_four_times_son son_age father_age) 
  (h2 : sum_of_ages_ten_years_ago son_age father_age) : 
  father_age = 64 :=
sorry

end NUMINAMATH_GPT_father_l491_49196


namespace NUMINAMATH_GPT_evaluate_expression_l491_49113

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (1/3 : ℚ)
  let z := (-12 : ℚ)
  let w := (5 : ℚ)
  x^2 * y^3 * z + w = (179/36 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l491_49113


namespace NUMINAMATH_GPT_sum_of_circle_center_coordinates_l491_49167

open Real

theorem sum_of_circle_center_coordinates :
  let x1 := 5
  let y1 := 3
  let x2 := -7
  let y2 := 9
  let x_m := (x1 + x2) / 2
  let y_m := (y1 + y2) / 2
  x_m + y_m = 5 := by
  sorry

end NUMINAMATH_GPT_sum_of_circle_center_coordinates_l491_49167


namespace NUMINAMATH_GPT_prob_red_blue_calc_l491_49151

noncomputable def prob_red_blue : ℚ :=
  let p_yellow := (6 : ℚ) / 13
  let p_red_blue_given_yellow := (7 : ℚ) / 12
  let p_red_blue_given_not_yellow := (7 : ℚ) / 13
  p_red_blue_given_yellow * p_yellow + p_red_blue_given_not_yellow * (1 - p_yellow)

/-- The probability of drawing a red or blue marble from the updated bag contents is 91/169. -/
theorem prob_red_blue_calc : prob_red_blue = 91 / 169 :=
by
  -- This proof is omitted as per instructions.
  sorry

end NUMINAMATH_GPT_prob_red_blue_calc_l491_49151


namespace NUMINAMATH_GPT_min_value_frac_sum_l491_49198

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (3 * z / (x + 2 * y) + 5 * x / (2 * y + 3 * z) + 2 * y / (3 * x + z)) ≥ 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_sum_l491_49198


namespace NUMINAMATH_GPT_age_proof_l491_49157

theorem age_proof (M S Y : ℕ) (h1 : M = 36) (h2 : S = 12) (h3 : M = 3 * S) : 
  (M + Y = 2 * (S + Y)) ↔ (Y = 12) :=
by 
  sorry

end NUMINAMATH_GPT_age_proof_l491_49157


namespace NUMINAMATH_GPT_perimeter_one_face_of_cube_is_24_l491_49122

noncomputable def cube_volume : ℝ := 216
def perimeter_of_face_of_cube (V : ℝ) : ℝ := 4 * (V^(1/3) : ℝ)

theorem perimeter_one_face_of_cube_is_24 :
  perimeter_of_face_of_cube cube_volume = 24 := 
by
  -- This proof will invoke the calculation shown in the problem.
  sorry

end NUMINAMATH_GPT_perimeter_one_face_of_cube_is_24_l491_49122


namespace NUMINAMATH_GPT_new_profit_is_220_percent_l491_49133

noncomputable def cost_price (CP : ℝ) : ℝ := 100

def initial_profit_percentage : ℝ := 60

noncomputable def initial_selling_price (CP : ℝ) : ℝ :=
  CP + (initial_profit_percentage / 100) * CP

noncomputable def new_selling_price (SP : ℝ) : ℝ :=
  2 * SP

noncomputable def new_profit_percentage (CP SP2 : ℝ) : ℝ :=
  ((SP2 - CP) / CP) * 100

theorem new_profit_is_220_percent : 
  new_profit_percentage (cost_price 100) (new_selling_price (initial_selling_price (cost_price 100))) = 220 :=
by
  sorry

end NUMINAMATH_GPT_new_profit_is_220_percent_l491_49133


namespace NUMINAMATH_GPT_maximize_x4y3_l491_49106

theorem maximize_x4y3 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = 40) : 
    (x, y) = (160 / 7, 120 / 7) ↔ x ^ 4 * y ^ 3 ≤ (160 / 7) ^ 4 * (120 / 7) ^ 3 := 
sorry

end NUMINAMATH_GPT_maximize_x4y3_l491_49106


namespace NUMINAMATH_GPT_find_constants_l491_49134

theorem find_constants :
  ∃ (A B C : ℝ), (∀ x : ℝ, x ≠ 3 → x ≠ 4 → 
  (6 * x / ((x - 4) * (x - 3) ^ 2)) = (A / (x - 4) + B / (x - 3) + C / (x - 3) ^ 2)) ∧
  A = 24 ∧
  B = - 162 / 7 ∧
  C = - 18 :=
by
  use 24, -162 / 7, -18
  sorry

end NUMINAMATH_GPT_find_constants_l491_49134


namespace NUMINAMATH_GPT_S_10_is_65_l491_49109

variable (a_1 d : ℤ)
variable (S : ℤ → ℤ)

-- Define the arithmetic sequence conditions
def a_3 : ℤ := a_1 + 2 * d
def S_n (n : ℤ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom a_3_is_4 : a_3 = 4
axiom S_9_minus_S_6_is_27 : S 9 - S 6 = 27

-- The target statement to be proven
theorem S_10_is_65 : S 10 = 65 :=
by
  sorry

end NUMINAMATH_GPT_S_10_is_65_l491_49109


namespace NUMINAMATH_GPT_dinner_cakes_today_6_l491_49130

-- Definitions based on conditions
def lunch_cakes_today : ℕ := 5
def dinner_cakes_today (x : ℕ) : ℕ := x
def yesterday_cakes : ℕ := 3
def total_cakes_served : ℕ := 14

-- Lean statement to prove the mathematical equivalence
theorem dinner_cakes_today_6 (x : ℕ) (h : lunch_cakes_today + dinner_cakes_today x + yesterday_cakes = total_cakes_served) : x = 6 :=
by {
  sorry -- Proof to be completed.
}

end NUMINAMATH_GPT_dinner_cakes_today_6_l491_49130


namespace NUMINAMATH_GPT_students_total_l491_49152

theorem students_total (position_eunjung : ℕ) (following_students : ℕ) (h1 : position_eunjung = 6) (h2 : following_students = 7) : 
  position_eunjung + following_students = 13 :=
by
  sorry

end NUMINAMATH_GPT_students_total_l491_49152


namespace NUMINAMATH_GPT_negation_of_proposition_l491_49114

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → (x + 1/x) ≥ 2

-- Define the negation of the original proposition
def negation_prop : Prop := ∃ x > 0, x + 1/x < 2

-- State that the negation of the original proposition is the stated negation
theorem negation_of_proposition : (¬ ∀ x, original_prop x) ↔ negation_prop := 
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l491_49114


namespace NUMINAMATH_GPT_triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l491_49128

theorem triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero
  (A B C : ℝ) (h : A + B + C = 180): 
    (A = 60 ∨ B = 60 ∨ C = 60) ↔ (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 0) := 
by
  sorry

end NUMINAMATH_GPT_triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l491_49128


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l491_49135

theorem largest_divisor_of_expression (x : ℤ) (h_even : x % 2 = 0) :
  ∃ k, (∀ x, x % 2 = 0 → k ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) ∧ 
       (∀ m, (∀ x, x % 2 = 0 → m ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) → m ≤ k) ∧ 
       k = 32 :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l491_49135


namespace NUMINAMATH_GPT_problem_proof_l491_49156

noncomputable def original_number_of_buses_and_total_passengers : Nat × Nat :=
  let k := 24
  let total_passengers := 529
  (k, total_passengers)

theorem problem_proof (k n : Nat) (h₁ : n = 22 + 23 / (k - 1)) (h₂ : 22 * k + 1 = n * (k - 1)) (h₃ : k ≥ 2) (h₄ : n ≤ 32) :
  (k, 22 * k + 1) = original_number_of_buses_and_total_passengers :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l491_49156


namespace NUMINAMATH_GPT_total_surface_area_correct_l491_49191

noncomputable def total_surface_area_of_cylinder (radius height : ℝ) : ℝ :=
  let lateral_surface_area := 2 * Real.pi * radius * height
  let top_and_bottom_area := 2 * Real.pi * radius^2
  lateral_surface_area + top_and_bottom_area

theorem total_surface_area_correct : total_surface_area_of_cylinder 3 10 = 78 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_correct_l491_49191


namespace NUMINAMATH_GPT_irreducible_fraction_l491_49105

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end NUMINAMATH_GPT_irreducible_fraction_l491_49105


namespace NUMINAMATH_GPT_power_function_value_at_two_l491_49118

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_value_at_two (a : ℝ) (h : f (1/2) a = 8) : f 2 a = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_power_function_value_at_two_l491_49118


namespace NUMINAMATH_GPT_smallest_digit_is_one_l491_49168

-- Given a 4-digit integer x.
def four_digit_integer (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

-- Define function for the product of digits of x.
def product_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 * d2 * d3 * d4

-- Define function for the sum of digits of x.
def sum_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 + d2 + d3 + d4

-- Assume p is a prime number.
def is_prime (p : ℕ) : Prop :=
  ¬ ∃ d, d ∣ p ∧ d ≠ 1 ∧ d ≠ p

-- Proof problem: Given conditions for T(x) and S(x),
-- prove that the smallest digit in x is 1.
theorem smallest_digit_is_one (x p k : ℕ) (h1 : four_digit_integer x)
  (h2 : is_prime p) (h3 : product_of_digits x = p^k)
  (h4 : sum_of_digits x = p^p - 5) : 
  ∃ d1 d2 d3 d4, d1 <= d2 ∧ d1 <= d3 ∧ d1 <= d4 ∧ d1 = 1 
  ∧ (d1 + d2 + d3 + d4 = p^p - 5) 
  ∧ (d1 * d2 * d3 * d4 = p^k) := 
sorry

end NUMINAMATH_GPT_smallest_digit_is_one_l491_49168


namespace NUMINAMATH_GPT_problem_l491_49146

variable (m n : ℝ)
variable (h1 : m + n = -1994)
variable (h2 : m * n = 7)

theorem problem (m n : ℝ) (h1 : m + n = -1994) (h2 : m * n = 7) : 
  (m^2 + 1993 * m + 6) * (n^2 + 1995 * n + 8) = 1986 := 
by
  sorry

end NUMINAMATH_GPT_problem_l491_49146


namespace NUMINAMATH_GPT_new_room_area_l491_49186

def holden_master_bedroom : Nat := 309
def holden_master_bathroom : Nat := 150

theorem new_room_area : 
  (holden_master_bedroom + holden_master_bathroom) * 2 = 918 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_new_room_area_l491_49186


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l491_49121

-- Definition of the conditions
def Q (x : ℝ) : Prop := x^2 - x - 2 > 0
def P (x a : ℝ) : Prop := |x| > a

-- Main statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, P x a → Q x) → a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l491_49121


namespace NUMINAMATH_GPT_gcd_123456_789012_l491_49161

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := sorry

end NUMINAMATH_GPT_gcd_123456_789012_l491_49161


namespace NUMINAMATH_GPT_difference_of_squares_is_40_l491_49115

theorem difference_of_squares_is_40 {x y : ℕ} (h1 : x + y = 20) (h2 : x * y = 99) (hx : x > y) : x^2 - y^2 = 40 :=
sorry

end NUMINAMATH_GPT_difference_of_squares_is_40_l491_49115


namespace NUMINAMATH_GPT_range_of_a_l491_49112

noncomputable section

def f (a x : ℝ) := a * x^2 + 2 * a * x - Real.log (x + 1)
def g (x : ℝ) := (Real.exp x - x - 1) / (Real.exp x * (x + 1))

theorem range_of_a
  (a : ℝ)
  (h : ∀ x > 0, f a x + Real.exp (-a) > 1 / (x + 1)) : a ∈ Set.Ici (1 / 2) := 
sorry

end NUMINAMATH_GPT_range_of_a_l491_49112


namespace NUMINAMATH_GPT_expected_number_of_different_faces_l491_49176

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end NUMINAMATH_GPT_expected_number_of_different_faces_l491_49176


namespace NUMINAMATH_GPT_cucumbers_count_l491_49145

theorem cucumbers_count:
  ∀ (C T : ℕ), C + T = 420 ∧ T = 4 * C → C = 84 :=
by
  intros C T h
  sorry

end NUMINAMATH_GPT_cucumbers_count_l491_49145


namespace NUMINAMATH_GPT_remainder_problem_l491_49153

theorem remainder_problem :
  (1234567 % 135 = 92) ∧ ((92 * 5) % 27 = 1) := by
  sorry

end NUMINAMATH_GPT_remainder_problem_l491_49153


namespace NUMINAMATH_GPT_new_perimeter_is_20_l491_49181

/-
Ten 1x1 square tiles are arranged to form a figure whose outside edges form a polygon with a perimeter of 16 units.
Four additional tiles of the same size are added to the figure so that each new tile shares at least one side with 
one of the squares in the original figure. Prove that the new perimeter of the figure could be 20 units.
-/

theorem new_perimeter_is_20 (initial_perimeter : ℕ) (num_initial_tiles : ℕ) 
                            (num_new_tiles : ℕ) (shared_sides : ℕ) 
                            (total_tiles : ℕ) : 
  initial_perimeter = 16 → num_initial_tiles = 10 → num_new_tiles = 4 → 
  shared_sides ≤ 8 → total_tiles = 14 → (initial_perimeter + 2 * (num_new_tiles - shared_sides)) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_new_perimeter_is_20_l491_49181


namespace NUMINAMATH_GPT_parallel_lines_intersection_value_of_c_l491_49192

theorem parallel_lines_intersection_value_of_c
  (a b c : ℝ) (h_parallel : a = -4 * b)
  (h1 : a * 2 - 2 * (-4) = c) (h2 : 2 * 2 + b * (-4) = c) :
  c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_parallel_lines_intersection_value_of_c_l491_49192
