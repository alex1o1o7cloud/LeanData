import Mathlib

namespace polygon_sides_l42_4211

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by sorry

end polygon_sides_l42_4211


namespace part1_part2_l42_4224

open Real

noncomputable def condition1 (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a^2 + 3 * b^2 = 3

theorem part1 {a b : ℝ} (h : condition1 a b) : sqrt 5 * a + b ≤ 4 := 
sorry

theorem part2 {x a b : ℝ} (h₁ : condition1 a b) (h₂ : 2 * abs (x - 1) + abs x ≥ 4) : 
x ≤ -2/3 ∨ x ≥ 2 := 
sorry

end part1_part2_l42_4224


namespace vincent_earnings_l42_4295

def fantasy_book_cost : ℕ := 6
def literature_book_cost : ℕ := fantasy_book_cost / 2
def mystery_book_cost : ℕ := 4

def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def mystery_books_sold_per_day : ℕ := 3

def daily_earnings : ℕ :=
  (fantasy_books_sold_per_day * fantasy_book_cost) +
  (literature_books_sold_per_day * literature_book_cost) +
  (mystery_books_sold_per_day * mystery_book_cost)

def total_earnings_after_seven_days : ℕ :=
  daily_earnings * 7

theorem vincent_earnings : total_earnings_after_seven_days = 462 :=
by
  sorry

end vincent_earnings_l42_4295


namespace proportional_b_value_l42_4291

theorem proportional_b_value (b : ℚ) : (∃ k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, x + 2 - 3 * b = k * x)) ↔ b = 2 / 3 :=
by
  sorry

end proportional_b_value_l42_4291


namespace total_texts_received_l42_4242

structure TextMessageScenario :=
  (textsBeforeNoon : Nat)
  (textsAtNoon : Nat)
  (textsAfterNoonDoubling : (Nat → Nat) → Nat)
  (textsAfter6pm : (Nat → Nat) → Nat)

def textsBeforeNoon := 21
def textsAtNoon := 2

-- Calculation for texts received from noon to 6 pm
def noonTo6pmTexts (textsAtNoon : Nat) : Nat :=
  let rec doubling (n : Nat) : Nat := match n with
    | 0 => textsAtNoon
    | n + 1 => 2 * (doubling n)
  (doubling 0) + (doubling 1) + (doubling 2) + (doubling 3) + (doubling 4) + (doubling 5)

def textsAfterNoonDoubling : (Nat → Nat) → Nat := λ doubling => noonTo6pmTexts 2

-- Calculation for texts received from 6 pm to midnight
def after6pmTexts (textsAt6pm : Nat) : Nat :=
  let rec decrease (n : Nat) : Nat := match n with
    | 0 => textsAt6pm
    | n + 1 => (decrease n) - 5
  (decrease 0) + (decrease 1) + (decrease 2) + (decrease 3) + (decrease 4) + (decrease 5) + (decrease 6)

def textsAfter6pm : (Nat → Nat) → Nat := λ decrease => after6pmTexts 64

theorem total_texts_received : textsBeforeNoon + (textsAfterNoonDoubling (λ x => x)) + (textsAfter6pm (λ x => x)) = 490 := by
  sorry
 
end total_texts_received_l42_4242


namespace unloading_time_relationship_l42_4235

-- Conditions
def loading_speed : ℝ := 30
def loading_time : ℝ := 8
def total_tonnage : ℝ := loading_speed * loading_time
def unloading_speed (x : ℝ) : ℝ := x

-- Proof statement
theorem unloading_time_relationship (x : ℝ) (hx : x ≠ 0) : 
  ∀ y : ℝ, y = 240 / x :=
by 
  sorry

end unloading_time_relationship_l42_4235


namespace average_speed_round_trip_l42_4268

/--
Let \( d = 150 \) miles be the distance from City \( X \) to City \( Y \).
Let \( v1 = 50 \) mph be the speed from \( X \) to \( Y \).
Let \( v2 = 30 \) mph be the speed from \( Y \) to \( X \).
Then the average speed for the round trip is 37.5 mph.
-/
theorem average_speed_round_trip :
  let d := 150
  let v1 := 50
  let v2 := 30
  (2 * d) / ((d / v1) + (d / v2)) = 37.5 :=
by
  sorry

end average_speed_round_trip_l42_4268


namespace reflection_slope_intercept_l42_4207

noncomputable def reflect_line_slope_intercept (k : ℝ) (hk1 : k ≠ 0) (hk2 : k ≠ -1) : ℝ × ℝ :=
  let slope := (1 : ℝ) / k
  let intercept := (k - 1) / k
  (slope, intercept)

theorem reflection_slope_intercept {k : ℝ} (hk1 : k ≠ 0) (hk2 : k ≠ -1) :
  reflect_line_slope_intercept k hk1 hk2 = (1/k, (k-1)/k) := by
  sorry

end reflection_slope_intercept_l42_4207


namespace JessieScore_l42_4281

-- Define the conditions as hypotheses
variables (correct_answers : ℕ) (incorrect_answers : ℕ) (unanswered_questions : ℕ)
variables (points_per_correct : ℕ) (points_deducted_per_incorrect : ℤ)

-- Define the values for the specific problem instance
def JessieCondition := correct_answers = 16 ∧ incorrect_answers = 4 ∧ unanswered_questions = 10 ∧
                       points_per_correct = 2 ∧ points_deducted_per_incorrect = -1 / 2

-- Define the statement that Jessie's score is 30 given the conditions
theorem JessieScore (h : JessieCondition correct_answers incorrect_answers unanswered_questions points_per_correct points_deducted_per_incorrect) :
  (correct_answers * points_per_correct : ℤ) + (incorrect_answers * points_deducted_per_incorrect) = 30 :=
by
  sorry

end JessieScore_l42_4281


namespace find_y_in_terms_of_x_l42_4250

variable (x y : ℝ)

theorem find_y_in_terms_of_x (hx : x = 5) (hy : y = -4) (hp : ∃ k, y = k * (x - 3)) :
  y = -2 * x + 6 := by
sorry

end find_y_in_terms_of_x_l42_4250


namespace smallest_even_n_for_reducible_fraction_l42_4279

theorem smallest_even_n_for_reducible_fraction : 
  ∃ (N: ℕ), (N > 2013) ∧ (N % 2 = 0) ∧ (Nat.gcd (15 * N - 7) (22 * N - 5) > 1) ∧ N = 2144 :=
sorry

end smallest_even_n_for_reducible_fraction_l42_4279


namespace total_cost_l42_4246

-- Defining the prices based on the given conditions
def price_smartphone : ℕ := 300
def price_pc : ℕ := price_smartphone + 500
def price_tablet : ℕ := price_smartphone + price_pc

-- The theorem to prove the total cost of buying one of each product
theorem total_cost : price_smartphone + price_pc + price_tablet = 2200 :=
by
  sorry

end total_cost_l42_4246


namespace c_plus_d_is_even_l42_4263

-- Define the conditions
variables {c d : ℕ}
variables (m n : ℕ) (hc : c = 6 * m) (hd : d = 9 * n)

-- State the theorem to be proven
theorem c_plus_d_is_even : 
  (c = 6 * m) → (d = 9 * n) → Even (c + d) :=
by
  -- Proof steps would go here
  sorry

end c_plus_d_is_even_l42_4263


namespace infinite_solutions_l42_4233

theorem infinite_solutions (x : ℕ) :
  15 < 2 * x + 10 ↔ ∃ n : ℕ, x = n + 3 :=
by {
  sorry
}

end infinite_solutions_l42_4233


namespace max_sundays_in_84_days_l42_4229

-- Define constants
def days_in_week : ℕ := 7
def total_days : ℕ := 84

-- Theorem statement
theorem max_sundays_in_84_days : (total_days / days_in_week) = 12 :=
by sorry

end max_sundays_in_84_days_l42_4229


namespace num_articles_cost_price_l42_4205

theorem num_articles_cost_price (N C S : ℝ) (h1 : N * C = 50 * S) (h2 : (S - C) / C * 100 = 10) : N = 55 := 
sorry

end num_articles_cost_price_l42_4205


namespace abs_diff_between_sequences_l42_4239

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 + (-8) * (n - 1)

theorem abs_diff_between_sequences :
  |sequence_C 31 - sequence_D 31| = 600 :=
by
  sorry

end abs_diff_between_sequences_l42_4239


namespace first_shaded_square_in_each_column_l42_4204

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem first_shaded_square_in_each_column : 
  ∃ n, triangular_number n = 120 ∧ ∀ m < n, ¬ ∀ k < 8, ∃ j ≤ m, ((triangular_number j) % 8) = k := 
by
  sorry

end first_shaded_square_in_each_column_l42_4204


namespace consecutive_sum_impossible_l42_4283

theorem consecutive_sum_impossible (n : ℕ) :
  (¬ (∃ (a b : ℕ), a < b ∧ n = (b - a + 1) * (a + b) / 2)) ↔ ∃ s : ℕ, n = 2 ^ s :=
sorry

end consecutive_sum_impossible_l42_4283


namespace circle_reflection_l42_4290

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l42_4290


namespace sum_of_squares_l42_4221

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end sum_of_squares_l42_4221


namespace max_area_of_triangle_l42_4203

theorem max_area_of_triangle (AB AC BC : ℝ) : 
  AB = 4 → AC = 2 * BC → 
  ∃ (S : ℝ), (∀ (S' : ℝ), S' ≤ S) ∧ S = 16 / 3 :=
by
  sorry

end max_area_of_triangle_l42_4203


namespace Antoinette_weight_l42_4270

-- Define weights for Antoinette and Rupert
variables (A R : ℕ)

-- Define the given conditions
def condition1 := A = 2 * R - 7
def condition2 := A + R = 98

-- The theorem to prove under the given conditions
theorem Antoinette_weight : condition1 A R → condition2 A R → A = 63 := 
by {
  -- The proof is omitted
  sorry
}

end Antoinette_weight_l42_4270


namespace fewest_apples_l42_4248

-- Definitions based on the conditions
def Yoongi_apples : Nat := 4
def Jungkook_initial_apples : Nat := 6
def Jungkook_additional_apples : Nat := 3
def Jungkook_apples : Nat := Jungkook_initial_apples + Jungkook_additional_apples
def Yuna_apples : Nat := 5

-- Main theorem based on the question and the correct answer
theorem fewest_apples : Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end fewest_apples_l42_4248


namespace longest_side_of_triangle_l42_4274

-- Defining variables and constants
variables (x : ℕ)

-- Defining the side lengths of the triangle
def side1 := 7
def side2 := x + 4
def side3 := 2 * x + 1

-- Defining the perimeter of the triangle
def perimeter := side1 + side2 + side3

-- Statement of the main theorem
theorem longest_side_of_triangle (h : perimeter x = 36) : max side1 (max (side2 x) (side3 x)) = 17 :=
by sorry

end longest_side_of_triangle_l42_4274


namespace tom_gave_2_seashells_to_jessica_l42_4288

-- Conditions
def original_seashells : Nat := 5
def current_seashells : Nat := 3

-- Question as a proposition
def seashells_given (x : Nat) : Prop :=
  original_seashells - current_seashells = x

-- The proof problem
theorem tom_gave_2_seashells_to_jessica : seashells_given 2 :=
by 
  sorry

end tom_gave_2_seashells_to_jessica_l42_4288


namespace cos_at_min_distance_l42_4222

noncomputable def cosAtMinimumDistance (t : ℝ) (ht : t < 0) : ℝ :=
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  if distance = Real.sqrt 5 then
    x / distance
  else
    0 -- some default value given the condition distance is not sqrt(5), which is impossible in this context

theorem cos_at_min_distance (t : ℝ) (ht : t < 0) :
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  distance = Real.sqrt 5 → cosAtMinimumDistance t ht = - 2 * Real.sqrt 5 / 5 :=
by
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  sorry

end cos_at_min_distance_l42_4222


namespace river_width_l42_4264

theorem river_width
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) (flow_rate_m_per_min : ℝ)
  (H_depth : depth = 5)
  (H_flow_rate_kmph : flow_rate_kmph = 4)
  (H_volume_per_minute : volume_per_minute = 6333.333333333333)
  (H_flow_rate_m_per_min : flow_rate_m_per_min = 66.66666666666667) :
  volume_per_minute / (depth * flow_rate_m_per_min) = 19 :=
by
  -- proof goes here
  sorry

end river_width_l42_4264


namespace greatest_cars_with_ac_not_racing_stripes_l42_4261

-- Definitions
def total_cars : ℕ := 100
def cars_without_ac : ℕ := 47
def cars_with_ac : ℕ := total_cars - cars_without_ac
def at_least_racing_stripes : ℕ := 53

-- Prove that the greatest number of cars that could have air conditioning but not racing stripes is 53
theorem greatest_cars_with_ac_not_racing_stripes :
  ∃ maximum_cars_with_ac_not_racing_stripes, 
    maximum_cars_with_ac_not_racing_stripes = cars_with_ac - 0 ∧
    maximum_cars_with_ac_not_racing_stripes = 53 := 
by
  sorry

end greatest_cars_with_ac_not_racing_stripes_l42_4261


namespace mart_income_percentage_l42_4206

variables (T J M : ℝ)

theorem mart_income_percentage (h1 : M = 1.60 * T) (h2 : T = 0.50 * J) :
  M = 0.80 * J :=
by
  sorry

end mart_income_percentage_l42_4206


namespace sum_of_squares_of_consecutive_integers_divisible_by_5_l42_4244

theorem sum_of_squares_of_consecutive_integers_divisible_by_5 (n : ℤ) :
  (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_divisible_by_5_l42_4244


namespace score_of_juniors_correct_l42_4275

-- Let the total number of students be 20
def total_students : ℕ := 20

-- 20% of the students are juniors
def juniors_percent : ℝ := 0.20

-- Total number of juniors
def number_of_juniors : ℕ := 4 -- 20% of 20

-- The remaining are seniors
def number_of_seniors : ℕ := 16 -- 80% of 20

-- Overall average score of all students
def overall_average_score : ℝ := 85

-- Average score of the seniors
def seniors_average_score : ℝ := 84

-- Calculate the total score of all students
def total_score : ℝ := overall_average_score * total_students

-- Calculate the total score of the seniors
def total_score_of_seniors : ℝ := seniors_average_score * number_of_seniors

-- We need to prove that the score of each junior
def score_of_each_junior : ℝ := 89

theorem score_of_juniors_correct :
  (total_score - total_score_of_seniors) / number_of_juniors = score_of_each_junior :=
by
  sorry

end score_of_juniors_correct_l42_4275


namespace train_speed_kmph_l42_4276

theorem train_speed_kmph (length time : ℝ) (h_length : length = 90) (h_time : time = 8.999280057595392) :
  (length / time) * 3.6 = 36.003 :=
by
  rw [h_length, h_time]
  norm_num
  sorry -- the norm_num tactic might simplify this enough, otherwise further steps would be added here.

end train_speed_kmph_l42_4276


namespace find_smallest_number_l42_4231

theorem find_smallest_number
  (a1 a2 a3 a4 : ℕ)
  (h1 : (a1 + a2 + a3 + a4) / 4 = 30)
  (h2 : a2 = 28)
  (h3 : a2 = 35 - 7) :
  a1 = 27 :=
sorry

end find_smallest_number_l42_4231


namespace truck_loading_time_l42_4258

theorem truck_loading_time :
  let worker1_rate := (1:ℝ) / 6
  let worker2_rate := (1:ℝ) / 5
  let combined_rate := worker1_rate + worker2_rate
  (combined_rate != 0) → 
  (1 / combined_rate = (30:ℝ) / 11) :=
by
  sorry

end truck_loading_time_l42_4258


namespace total_toys_l42_4259

theorem total_toys (toys_kamari : ℕ) (toys_anais : ℕ) (h1 : toys_kamari = 65) (h2 : toys_anais = toys_kamari + 30) :
  toys_kamari + toys_anais = 160 :=
by 
  sorry

end total_toys_l42_4259


namespace f_2016_eq_one_third_l42_4252

noncomputable def f (x : ℕ) : ℝ := sorry

axiom f_one : f 1 = 2
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = (1 + f x) / (1 - f x)

theorem f_2016_eq_one_third : f 2016 = 1 / 3 := sorry

end f_2016_eq_one_third_l42_4252


namespace austin_more_apples_than_dallas_l42_4284

-- Conditions as definitions
def dallas_apples : ℕ := 14
def dallas_pears : ℕ := 9
def austin_pears : ℕ := dallas_pears - 5
def austin_total_fruit : ℕ := 24

-- The theorem statement
theorem austin_more_apples_than_dallas 
  (austin_apples : ℕ) (h1 : austin_apples + austin_pears = austin_total_fruit) :
  austin_apples - dallas_apples = 6 :=
sorry

end austin_more_apples_than_dallas_l42_4284


namespace quadrilateral_area_is_114_5_l42_4241

noncomputable def area_of_quadrilateral_114_5 
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) : ℝ :=
  114.5

theorem quadrilateral_area_is_114_5
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) :
  area_of_quadrilateral_114_5 AB BC CD AD angle_ABC h1 h2 h3 h4 h5 = 114.5 :=
sorry

end quadrilateral_area_is_114_5_l42_4241


namespace tg_ctg_sum_l42_4201

theorem tg_ctg_sum (x : Real) 
  (h : Real.cos x ≠ 0 ∧ Real.sin x ≠ 0 ∧ 1 / Real.cos x - 1 / Real.sin x = 4 * Real.sqrt 3) :
  (Real.sin x / Real.cos x + Real.cos x / Real.sin x = 8 ∨ Real.sin x / Real.cos x + Real.cos x / Real.sin x = -6) :=
sorry

end tg_ctg_sum_l42_4201


namespace max_product_areas_l42_4267

theorem max_product_areas (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h : a + b + c + d = 1) :
  a * b * c * d ≤ 1 / 256 :=
sorry

end max_product_areas_l42_4267


namespace midpoint_of_AB_l42_4294

theorem midpoint_of_AB (xA xB : ℝ) (p : ℝ) (h_parabola : ∀ y, y^2 = 4 * xA → y^2 = 4 * xB)
  (h_focus : (2 : ℝ) = p)
  (h_length_AB : (abs (xB - xA)) = 5) :
  (xA + xB) / 2 = 3 / 2 :=
sorry

end midpoint_of_AB_l42_4294


namespace total_emails_675_l42_4286

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end total_emails_675_l42_4286


namespace marble_prob_l42_4260

theorem marble_prob (a c x y p q : ℕ) (h1 : 2 * a + c = 36) 
    (h2 : (x / a) * (x / a) * (y / c) = 1 / 3) 
    (h3 : (a - x) / a * (a - x) / a * (c - y) / c = p / q) 
    (hpq_rel_prime : Nat.gcd p q = 1) : p + q = 65 := by
  sorry

end marble_prob_l42_4260


namespace opposite_of_2023_l42_4282

/-- The opposite of a number n is defined as the number that, when added to n, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  sorry

end opposite_of_2023_l42_4282


namespace ratio_of_girls_l42_4230

theorem ratio_of_girls (total_julian_friends : ℕ) (percent_julian_girls : ℚ)
  (percent_julian_boys : ℚ) (total_boyd_friends : ℕ) (percent_boyd_boys : ℚ) :
  total_julian_friends = 80 →
  percent_julian_girls = 0.40 →
  percent_julian_boys = 0.60 →
  total_boyd_friends = 100 →
  percent_boyd_boys = 0.36 →
  (0.64 * total_boyd_friends : ℚ) / (0.40 * total_julian_friends : ℚ) = 2 :=
by
  sorry

end ratio_of_girls_l42_4230


namespace star_point_angle_l42_4202

theorem star_point_angle (n : ℕ) (h : n > 4) (h₁ : n ≥ 3) :
  ∃ θ : ℝ, θ = (n-2) * 180 / n :=
by
  sorry

end star_point_angle_l42_4202


namespace product_of_possible_x_values_l42_4289

theorem product_of_possible_x_values : 
  (∃ x1 x2 : ℚ, 
    (|15 / x1 + 4| = 3 ∧ |15 / x2 + 4| = 3) ∧
    -15 * -(15 / 7) = (225 / 7)) :=
sorry

end product_of_possible_x_values_l42_4289


namespace beef_stew_duration_l42_4251

noncomputable def original_portions : ℕ := 14
noncomputable def your_portion : ℕ := 1
noncomputable def roommate_portion : ℕ := 3
noncomputable def guest_portion : ℕ := 4
noncomputable def total_daily_consumption : ℕ := your_portion + roommate_portion + guest_portion
noncomputable def days_stew_lasts : ℕ := original_portions / total_daily_consumption

theorem beef_stew_duration : days_stew_lasts = 2 :=
by
  sorry

end beef_stew_duration_l42_4251


namespace range_of_expression_l42_4214

noncomputable def f (x : ℝ) := |Real.log x / Real.log 2|

theorem range_of_expression (a b : ℝ) (h_f_eq : f a = f b) (h_a_lt_b : a < b) :
  f a = f b → a < b → (∃ c > 3, c = (2 / a) + (1 / b)) := by
  sorry

end range_of_expression_l42_4214


namespace percent_motorists_exceeding_speed_limit_l42_4228

-- Definitions based on conditions:
def total_motorists := 100
def percent_receiving_tickets := 10
def percent_exceeding_no_ticket := 50

-- The Lean 4 statement to prove the question
theorem percent_motorists_exceeding_speed_limit :
  (percent_receiving_tickets + (percent_receiving_tickets * percent_exceeding_no_ticket / 100)) = 20 :=
by
  sorry

end percent_motorists_exceeding_speed_limit_l42_4228


namespace multiple_of_1984_exists_l42_4292

theorem multiple_of_1984_exists (a : Fin 97 → ℕ) (h_distinct: Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
  1984 ∣ (a i - a j) * (a k - a l) :=
by
  sorry

end multiple_of_1984_exists_l42_4292


namespace company_workers_l42_4297

theorem company_workers (W : ℕ) (H1 : (1/3 : ℚ) * W = ((1/3 : ℚ) * W)) 
  (H2 : 0.20 * ((1/3 : ℚ) * W) = ((1/15 : ℚ) * W)) 
  (H3 : 0.40 * ((2/3 : ℚ) * W) = ((4/15 : ℚ) * W)) 
  (H4 : (4/15 : ℚ) * W + (4/15 : ℚ) * W = 160)
  : (W - 160 = 140) :=
by
  sorry

end company_workers_l42_4297


namespace lcm_18_24_30_eq_360_l42_4277

-- Define the three numbers in the condition
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 30

-- State the theorem to prove
theorem lcm_18_24_30_eq_360 : Nat.lcm a (Nat.lcm b c) = 360 :=
by 
  sorry -- Proof is omitted as per instructions

end lcm_18_24_30_eq_360_l42_4277


namespace part1_l42_4220

noncomputable def f (a x : ℝ) : ℝ := a * x - 2 * Real.log x + 2 * (1 + a) + (a - 2) / x

theorem part1 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 1 ≤ x → f a x ≥ 0) ↔ 1 ≤ a :=
sorry

end part1_l42_4220


namespace decreasing_interval_l42_4271

noncomputable def f (x : ℝ) := x^2 * Real.exp x

theorem decreasing_interval : ∀ x : ℝ, x > -2 ∧ x < 0 → deriv f x < 0 := 
by
  intro x h
  sorry

end decreasing_interval_l42_4271


namespace find_power_l42_4255

theorem find_power (some_power : ℕ) (k : ℕ) :
  k = 8 → (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → some_power = 16 :=
by
  intro h1 h2
  rw [h1] at h2
  sorry

end find_power_l42_4255


namespace bread_slices_per_loaf_l42_4299

theorem bread_slices_per_loaf (friends: ℕ) (total_loaves : ℕ) (slices_per_friend: ℕ) (total_slices: ℕ)
  (h1 : friends = 10) (h2 : total_loaves = 4) (h3 : slices_per_friend = 6) (h4 : total_slices = friends * slices_per_friend):
  total_slices / total_loaves = 15 :=
by
  sorry

end bread_slices_per_loaf_l42_4299


namespace find_numbers_l42_4232

theorem find_numbers (x y : ℝ) (h₁ : x + y = x * y) (h₂ : x * y = x / y) :
  (x = 1 / 2) ∧ (y = -1) := by
  sorry

end find_numbers_l42_4232


namespace sum_of_legs_eq_40_l42_4253

theorem sum_of_legs_eq_40
  (x : ℝ)
  (h1 : x > 0)
  (h2 : x^2 + (x + 2)^2 = 29^2) :
  x + (x + 2) = 40 :=
by
  sorry

end sum_of_legs_eq_40_l42_4253


namespace cost_of_agricultural_equipment_max_units_of_type_A_l42_4208

-- Define cost equations
variables (x y : ℝ)

-- Define conditions as hypotheses
def condition1 : Prop := 2 * x + y = 4.2
def condition2 : Prop := x + 3 * y = 5.1

-- Prove the costs are respectively 1.5 and 1.2
theorem cost_of_agricultural_equipment (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 1.5 ∧ y = 1.2 := sorry

-- Define the maximum units constraint
def total_cost (m : ℕ) : ℝ := 1.5 * m + 1.2 * (2 * m - 3)

-- Prove the maximum units of type A is 3
theorem max_units_of_type_A (m : ℕ) (h : total_cost m ≤ 10) : m ≤ 3 := sorry

end cost_of_agricultural_equipment_max_units_of_type_A_l42_4208


namespace find_f_neg_two_l42_4227

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : 3 * f (1 / x) + (2 * f x) / x = x ^ 2

theorem find_f_neg_two : f (-2) = 67 / 20 :=
by
  sorry

end find_f_neg_two_l42_4227


namespace necessary_and_sufficient_condition_l42_4215

open Set

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 ^ 2}

noncomputable def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 ^ 2 + (p.2 - a) ^ 2 ≤ 1}

theorem necessary_and_sufficient_condition (a : ℝ) :
  N a ⊆ M ↔ a ≥ 5 / 4 := sorry

end necessary_and_sufficient_condition_l42_4215


namespace horner_eval_v3_at_minus4_l42_4238

def f (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

def horner_form (x : ℤ) : ℤ :=
  let a6 := 3
  let a5 := 5
  let a4 := 6
  let a3 := 79
  let a2 := -8
  let a1 := 35
  let a0 := 12
  let v := a6
  let v1 := v * x + a5
  let v2 := v1 * x + a4
  let v3 := v2 * x + a3
  let v4 := v3 * x + a2
  let v5 := v4 * x + a1
  let v6 := v5 * x + a0
  v3

theorem horner_eval_v3_at_minus4 :
  horner_form (-4) = -57 :=
by
  sorry

end horner_eval_v3_at_minus4_l42_4238


namespace calculate_salary_l42_4212

-- Define the constants and variables
def food_percentage : ℝ := 0.35
def rent_percentage : ℝ := 0.25
def clothes_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def recreational_percentage : ℝ := 0.15
def emergency_fund : ℝ := 3000
def total_percentage : ℝ := food_percentage + rent_percentage + clothes_percentage + transportation_percentage + recreational_percentage

-- Define the salary
def salary (S : ℝ) : Prop :=
  (total_percentage - 1) * S = emergency_fund

-- The theorem stating the salary is 60000
theorem calculate_salary : ∃ S : ℝ, salary S ∧ S = 60000 :=
by
  use 60000
  unfold salary total_percentage
  sorry

end calculate_salary_l42_4212


namespace arrangement_problem_l42_4247

def numWaysToArrangeParticipants : ℕ := 90

theorem arrangement_problem :
  ∃ (boys : ℕ) (girls : ℕ) (select_boys : ℕ → ℕ) (select_girls : ℕ → ℕ)
    (arrange : ℕ × ℕ × ℕ → ℕ),
  boys = 3 ∧ girls = 5 ∧
  select_boys boys = 3 ∧ select_girls girls = 5 ∧ 
  arrange (select_boys boys, select_girls girls, 2) = numWaysToArrangeParticipants :=
by
  sorry

end arrangement_problem_l42_4247


namespace inequality_solution_l42_4256

theorem inequality_solution :
  {x : ℝ | -x^2 - |x| + 6 > 0} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

end inequality_solution_l42_4256


namespace grandpa_max_pieces_l42_4265

theorem grandpa_max_pieces (m n : ℕ) (h : (m - 3) * (n - 3) = 9) : m * n = 112 :=
sorry

end grandpa_max_pieces_l42_4265


namespace angle_measure_l42_4213

variable (x : ℝ)

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem angle_measure (h : supplement x = 8 * complement x) : x = 540 / 7 := by
  sorry

end angle_measure_l42_4213


namespace jill_trips_to_fill_tank_l42_4273

-- Definitions as per the conditions specified
def tank_capacity : ℕ := 600
def bucket_capacity : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_trips_ratio : ℕ := 3
def jill_trips_ratio : ℕ := 2
def leak_per_trip : ℕ := 2

-- Prove that the number of trips Jill will make = 20 given the above conditions
theorem jill_trips_to_fill_tank : 
  (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) * (tank_capacity / ((jack_trips_ratio + jill_trips_ratio) * (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) / (jack_trips_ratio + jill_trips_ratio)))  = 20 := 
sorry

end jill_trips_to_fill_tank_l42_4273


namespace largest_divisor_same_remainder_l42_4240

theorem largest_divisor_same_remainder 
  (d : ℕ) (r : ℕ)
  (a b c : ℕ) 
  (h13511 : 13511 = a * d + r) 
  (h13903 : 13903 = b * d + r)
  (h14589 : 14589 = c * d + r) :
  d = 98 :=
by 
  sorry

end largest_divisor_same_remainder_l42_4240


namespace plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l42_4237

theorem plan_Y_cheaper_than_X (x : ℕ) : 
  ∃ x, 2500 + 7 * x < 15 * x ∧ ∀ y, y < x → ¬ (2500 + 7 * y < 15 * y) := 
sorry

theorem plan_Z_cheaper_than_X (x : ℕ) : 
  ∃ x, 3000 + 6 * x < 15 * x ∧ ∀ y, y < x → ¬ (3000 + 6 * y < 15 * y) := 
sorry

end plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l42_4237


namespace angle_C_in_triangle_l42_4298

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 90) (h2 : A + B + C = 180) : C = 90 :=
sorry

end angle_C_in_triangle_l42_4298


namespace barbara_total_cost_l42_4209

-- Define conditions
def steak_weight : ℝ := 4.5
def steak_price_per_pound : ℝ := 15.0
def chicken_weight : ℝ := 1.5
def chicken_price_per_pound : ℝ := 8.0

-- Define total cost formula
def total_cost := (steak_weight * steak_price_per_pound) + (chicken_weight * chicken_price_per_pound)

-- Prove that the total cost equals $79.50
theorem barbara_total_cost : total_cost = 79.50 := by
  sorry

end barbara_total_cost_l42_4209


namespace value_of_m_l42_4236

theorem value_of_m (m : ℝ) (h : m ≠ 0)
  (h_roots : ∀ x, m * x^2 + 8 * m * x + 60 = 0 ↔ x = -5 ∨ x = -3) :
  m = 4 :=
sorry

end value_of_m_l42_4236


namespace largest_non_prime_sum_l42_4243

theorem largest_non_prime_sum (a b n : ℕ) (h1 : a ≥ 1) (h2 : b < 47) (h3 : n = 47 * a + b) (h4 : ∀ b, b < 47 → ¬Nat.Prime b → b = 43) : 
  n = 90 :=
by
  sorry

end largest_non_prime_sum_l42_4243


namespace hyperbola_foci_problem_l42_4293

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

noncomputable def foci_1 : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def foci_2 : ℝ × ℝ := (Real.sqrt 5, 0)

noncomputable def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

noncomputable def vector (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v1.1 + v2.2 * v2.2

noncomputable def orthogonal (P : ℝ × ℝ) : Prop :=
  dot_product (vector P foci_1) (vector P foci_2) = 0

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def required_value (P : ℝ × ℝ) : ℝ :=
  distance P foci_1 * distance P foci_2

theorem hyperbola_foci_problem (P : ℝ × ℝ) : 
  point_on_hyperbola P → orthogonal P → required_value P = 2 := 
sorry

end hyperbola_foci_problem_l42_4293


namespace proof_of_problem_l42_4272

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, (x + 2) ^ (x + 3) = 1 ↔ (x = -1 ∨ x = -3)

theorem proof_of_problem : proof_problem :=
by
  sorry

end proof_of_problem_l42_4272


namespace smallest_k_value_for_screws_packs_l42_4234

theorem smallest_k_value_for_screws_packs :
  ∃ k : ℕ, k = 60 ∧ (∃ x y : ℕ, (k = 10 * x ∧ k = 12 * y) ∧ x ≠ y) := sorry

end smallest_k_value_for_screws_packs_l42_4234


namespace find_a_range_l42_4278

-- Definitions of sets A and B
def A (a x : ℝ) : Prop := a + 1 ≤ x ∧ x ≤ 2 * a - 1
def B (x : ℝ) : Prop := x ≤ 3 ∨ x > 5

-- Condition p: A ⊆ B
def p (a : ℝ) : Prop := ∀ x, A a x → B x

-- The function f(x) = x^2 - 2ax + 1
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Condition q: f(x) is increasing on (1/2, +∞)
def q (a : ℝ) : Prop := ∀ x y, 1/2 < x → x < y → f a x ≤ f a y

-- The given propositions
def prop1 (a : ℝ) : Prop := p a
def prop2 (a : ℝ) : Prop := q a

-- Given conditions
def given_conditions (a : ℝ) : Prop := ¬ (prop1 a ∧ prop2 a) ∧ (prop1 a ∨ prop2 a)

-- Proof statement: Find the range of values for 'a' according to the given conditions
theorem find_a_range (a : ℝ) :
  given_conditions a →
  (1/2 < a ∧ a ≤ 2) ∨ (4 < a) :=
sorry

end find_a_range_l42_4278


namespace score_order_l42_4269

variable (A B C D : ℕ)

theorem score_order (h1 : A + B = C + D) (h2 : C + A > B + D) (h3 : C > A + B) :
  (C > A ∧ A > B ∧ B > D) :=
by
  sorry

end score_order_l42_4269


namespace min_value_fraction_expr_l42_4254

theorem min_value_fraction_expr : ∀ (x : ℝ), x > 0 → (4 + x) * (1 + x) / x ≥ 9 :=
by
  sorry

end min_value_fraction_expr_l42_4254


namespace union_complement_correctness_l42_4280

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complement_correctness : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 4} →
  A ∪ (U \ B) = {1, 2, 3, 5} :=
by
  intro hU hA hB
  sorry

end union_complement_correctness_l42_4280


namespace conversion_base8_to_base10_l42_4245

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l42_4245


namespace base_conversion_l42_4216

theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 71) : x = 16 := 
by {
  sorry
}

end base_conversion_l42_4216


namespace clothing_weight_removed_l42_4218

/-- 
In a suitcase, the initial ratio of books to clothes to electronics, by weight measured in pounds, 
is 7:4:3. The electronics weight 9 pounds. Someone removes some pounds of clothing, doubling the ratio of books to clothes. 
This theorem verifies the weight of clothing removed is 1.5 pounds.
-/
theorem clothing_weight_removed 
  (B C E : ℕ) 
  (initial_ratio : B / 7 = C / 4 ∧ C / 4 = E / 3)
  (E_val : E = 9)
  (new_ratio : ∃ x : ℝ, B / (C - x) = 2) : 
  ∃ x : ℝ, x = 1.5 := 
sorry

end clothing_weight_removed_l42_4218


namespace isosceles_triangle_perimeter_l42_4257

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 4) (h2 : b = 9) (h3 : c = 9) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : a + b + c = 22 := 
by 
  sorry

end isosceles_triangle_perimeter_l42_4257


namespace how_many_times_faster_l42_4287

theorem how_many_times_faster (A B : ℝ) (h1 : A = 1 / 32) (h2 : A + B = 1 / 24) : A / B = 3 := by
  sorry

end how_many_times_faster_l42_4287


namespace initial_percentage_salt_l42_4226

theorem initial_percentage_salt :
  ∀ (P : ℝ),
  let Vi := 64 
  let Vf := 80
  let target_percent := 0.08
  (Vi * P = Vf * target_percent) → P = 0.1 :=
by
  intros P Vi Vf target_percent h
  have h1 : Vi = 64 := rfl
  have h2 : Vf = 80 := rfl
  have h3 : target_percent = 0.08 := rfl
  rw [h1, h2, h3] at h
  sorry

end initial_percentage_salt_l42_4226


namespace solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l42_4219

variable (a : ℝ) (x : ℝ)

def inequality (a x : ℝ) : Prop :=
  a * x^2 - (a + 2) * x + 2 < 0

theorem solve_inequality_when_a_lt_2 (h : a < 2) :
  (a = 0 → ∀ x, x > 1 → inequality a x) ∧
  (a < 0 → ∀ x, x < 2 / a ∨ x > 1 → inequality a x) ∧
  (0 < a ∧ a < 2 → ∀ x, 1 < x ∧ x < 2 / a → inequality a x) := 
sorry

theorem find_a_range_when_x_in_2_3 :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → inequality a x) → a < 2 / 3 :=
sorry

end solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l42_4219


namespace game_returns_to_A_after_three_rolls_l42_4296

theorem game_returns_to_A_after_three_rolls :
  (∃ i j k : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ 1 ≤ k ∧ k ≤ 6 ∧ (i + j + k) % 12 = 0) → 
  true :=
by
  sorry

end game_returns_to_A_after_three_rolls_l42_4296


namespace max_value_is_one_sixteenth_l42_4266

noncomputable def max_value_expression (t : ℝ) : ℝ :=
  (3^t - 4 * t) * t / 9^t

theorem max_value_is_one_sixteenth : 
  ∃ t : ℝ, max_value_expression t = 1 / 16 :=
sorry

end max_value_is_one_sixteenth_l42_4266


namespace x1_sufficient_not_necessary_l42_4200

theorem x1_sufficient_not_necessary : (x : ℝ) → (x = 1 ↔ (x - 1) * (x + 2) = 0) ∧ ∀ x, (x = 1 ∨ x = -2) → (x - 1) * (x + 2) = 0 ∧ (∀ y, (y - 1) * (y + 2) = 0 → (y = 1 ∨ y = -2)) :=
by
  sorry

end x1_sufficient_not_necessary_l42_4200


namespace quadratic_value_at_6_l42_4262

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 3

theorem quadratic_value_at_6 
  (a b : ℝ) (h : a ≠ 0) 
  (h_eq : f a b 2 = f a b 4) : 
  f a b 6 = -3 :=
by
  sorry

end quadratic_value_at_6_l42_4262


namespace carol_blocks_l42_4285

theorem carol_blocks (x : ℕ) (h : x - 25 = 17) : x = 42 :=
sorry

end carol_blocks_l42_4285


namespace square_perimeter_equals_66_88_l42_4210

noncomputable def circle_perimeter : ℝ := 52.5

noncomputable def circle_radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def circle_diameter (r : ℝ) : ℝ := 2 * r

noncomputable def square_side_length (d : ℝ) : ℝ := d

noncomputable def square_perimeter (s : ℝ) : ℝ := 4 * s

theorem square_perimeter_equals_66_88 :
  square_perimeter (square_side_length (circle_diameter (circle_radius circle_perimeter))) = 66.88 := 
by
  -- Placeholder for the proof
  sorry

end square_perimeter_equals_66_88_l42_4210


namespace find_x_for_set_6_l42_4249

theorem find_x_for_set_6 (x : ℝ) (h : 6 ∈ ({2, 4, x^2 - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end find_x_for_set_6_l42_4249


namespace equal_frac_implies_x_zero_l42_4217

theorem equal_frac_implies_x_zero (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
sorry

end equal_frac_implies_x_zero_l42_4217


namespace no_such_function_exists_l42_4223

open Classical

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (f 0 > 0) ∧ (∀ (x y : ℝ), f (x + y) ≥ f x + y * f (f x)) :=
sorry

end no_such_function_exists_l42_4223


namespace correct_value_l42_4225

-- Given condition
def incorrect_calculation (x : ℝ) : Prop := (x + 12) / 8 = 8

-- Theorem to prove the correct value
theorem correct_value (x : ℝ) (h : incorrect_calculation x) : (x - 12) * 9 = 360 :=
by
  sorry

end correct_value_l42_4225
